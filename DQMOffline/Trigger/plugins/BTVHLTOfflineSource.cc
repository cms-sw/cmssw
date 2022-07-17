/*
  BTVHLTOffline DQM code
*/
//
// Originally created by:  Anne-Catherine Le Bihan
//                         June 2015
//                         John Alison <johnalison@cmu.edu>
//                         June 2020
// Following the structure used in JetMetHLTOfflineSource

// system include files
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>
#include <cmath>
#include <iostream>

// user include files
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMOffline/Trigger/plugins/TriggerDQMBase.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/VertexReco/interface/Vertex.h"
#include "DataFormats/BTauReco/interface/JetTag.h"
#include "DataFormats/BTauReco/interface/ShallowTagInfo.h"
#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Common/interface/TriggerNames.h"
#include "CommonTools/UtilAlgos/interface/DeltaR.h"
#include "DataFormats/VertexReco/interface/VertexFwd.h"
#include "DataFormats/BTauReco/interface/SecondaryVertexTagInfo.h"
#include "DataFormats/TrackReco/interface/Track.h"

#include "TMath.h"
#include "TPRegexp.h"

class BTVHLTOfflineSource : public DQMEDAnalyzer {
public:
  explicit BTVHLTOfflineSource(const edm::ParameterSet&);
  ~BTVHLTOfflineSource() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;

  std::vector<const reco::Track*> getOfflineBTagTracks(float hltJetEta,
                                                       float hltJetPhi,
                                                       edm::Handle<edm::View<reco::BaseTagInfo>> offlineIPTagHandle,
                                                       std::vector<float>& offlineIP3D,
                                                       std::vector<float>& offlineIP3DSig);

  typedef reco::TemplatedSecondaryVertexTagInfo<reco::CandIPTagInfo, reco::VertexCompositePtrCandidate> SVTagInfo;

  template <class Base>
  std::vector<const reco::Track*> getOnlineBTagTracks(float hltJetEta,
                                                      float hltJetPhi,
                                                      edm::Handle<std::vector<Base>> jetSVTagsColl,
                                                      std::vector<float>& onlineIP3D,
                                                      std::vector<float>& onlineIP3DSig);

  void bookHistograms(DQMStore::IBooker&, edm::Run const& run, edm::EventSetup const& c) override;

  void dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) override;

  std::string dirname_;
  std::string processname_;
  bool verbose_;

  std::vector<std::pair<std::string, std::string>> custompathnamepairs_;

  edm::InputTag triggerSummaryLabel_;
  edm::InputTag triggerResultsLabel_;

  float turnon_threshold_loose_;
  float turnon_threshold_medium_;
  float turnon_threshold_tight_;

  edm::EDGetTokenT<reco::JetTagCollection> offlineDiscrTokenb_;
  edm::EDGetTokenT<reco::JetTagCollection> offlineDiscrTokenbb_;
  edm::EDGetTokenT<edm::View<reco::BaseTagInfo>> offlineIPToken_;

  edm::EDGetTokenT<std::vector<reco::Vertex>> hltFastPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> hltPFPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> hltCaloPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex>> offlinePVToken_;

  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsFUToken;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerSummaryToken;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerSummaryFUToken;

  edm::EDGetTokenT<std::vector<reco::ShallowTagInfo>> shallowTagInfosTokenCalo_;
  edm::EDGetTokenT<std::vector<reco::ShallowTagInfo>> shallowTagInfosTokenPf_;

  edm::EDGetTokenT<std::vector<reco::SecondaryVertexTagInfo>> SVTagInfosTokenCalo_;
  edm::EDGetTokenT<std::vector<SVTagInfo>> SVTagInfosTokenPf_;

  edm::EDGetTokenT<reco::JetTagCollection> caloTagsToken_;
  edm::EDGetTokenT<reco::JetTagCollection> pfTagsToken_;
  edm::Handle<reco::JetTagCollection> caloTags;
  edm::Handle<reco::JetTagCollection> pfTags;

  float minDecayLength_;
  float maxDecayLength_;
  float minJetDistance_;
  float maxJetDistance_;
  float dRTrackMatch_;

  HLTConfigProvider hltConfig_;

  class PathInfo : public TriggerDQMBase {
  public:
    PathInfo()
        : prescaleUsed_(-1),
          pathName_("unset"),
          filterName_("unset"),
          processName_("unset"),
          objectType_(-1),
          triggerType_("unset") {}

    ~PathInfo() override = default;

    PathInfo(const int prescaleUsed,
             const std::string& pathName,
             const std::string& filterName,
             const std::string& processName,
             const int type,
             const std::string& triggerType)
        : prescaleUsed_(prescaleUsed),
          pathName_(pathName),
          filterName_(filterName),
          processName_(processName),
          objectType_(type),
          triggerType_(triggerType) {}

    const std::string getLabel() const { return filterName_; }
    void setLabel(std::string labelName) { filterName_ = std::move(labelName); }
    const std::string getPath() const { return pathName_; }
    const int getprescaleUsed() const { return prescaleUsed_; }
    const std::string getProcess() const { return processName_; }
    const int getObjectType() const { return objectType_; }
    const std::string getTriggerType() const { return triggerType_; }
    const edm::InputTag getTag() const { return edm::InputTag(filterName_, "", processName_); }
    const bool operator==(const std::string& v) const { return v == pathName_; }

    MonitorElement* Discr = nullptr;
    MonitorElement* Pt = nullptr;
    MonitorElement* Eta = nullptr;
    MonitorElement* Discr_HLTvsRECO = nullptr;
    MonitorElement* Discr_HLTMinusRECO = nullptr;
    ObjME Discr_turnon_loose;
    ObjME Discr_turnon_medium;
    ObjME Discr_turnon_tight;
    MonitorElement* PVz = nullptr;
    MonitorElement* fastPVz = nullptr;
    MonitorElement* PVz_HLTMinusRECO = nullptr;
    MonitorElement* fastPVz_HLTMinusRECO = nullptr;
    MonitorElement* n_vtx = nullptr;
    MonitorElement* vtx_mass = nullptr;
    MonitorElement* n_vtx_trks = nullptr;
    MonitorElement* n_sel_tracks = nullptr;
    MonitorElement* h_3d_ip_distance = nullptr;
    MonitorElement* h_3d_ip_error = nullptr;
    MonitorElement* h_3d_ip_sig = nullptr;

    //NEW
    MonitorElement* h_jetNSecondaryVertices = nullptr;
    MonitorElement* h_jet_pt = nullptr;
    MonitorElement* h_jet_eta = nullptr;
    MonitorElement* h_trackSumJetEtRatio = nullptr;
    MonitorElement* h_trackSip2dValAboveCharm = nullptr;
    MonitorElement* h_trackSip2dSigAboveCharm = nullptr;
    MonitorElement* h_trackSip3dValAboveCharm = nullptr;
    MonitorElement* h_trackSip3dSigAboveCharm = nullptr;
    MonitorElement* h_jetNSelectedTracks = nullptr;
    MonitorElement* h_jetNTracksEtaRel = nullptr;
    MonitorElement* h_vertexCategory = nullptr;
    MonitorElement* h_trackSumJetDeltaR = nullptr;

    MonitorElement* h_trackJetDistVal = nullptr;
    MonitorElement* h_trackPtRel = nullptr;
    MonitorElement* h_trackDeltaR = nullptr;
    MonitorElement* h_trackPtRatio = nullptr;
    MonitorElement* h_trackSip3dSig = nullptr;
    MonitorElement* h_trackSip2dSig = nullptr;
    MonitorElement* h_trackDecayLenVal = nullptr;
    MonitorElement* h_trackEtaRel = nullptr;

    MonitorElement* h_vertexEnergyRatio = nullptr;
    MonitorElement* h_vertexJetDeltaR = nullptr;
    MonitorElement* h_flightDistance2dVal = nullptr;
    MonitorElement* h_flightDistance2dSig = nullptr;
    MonitorElement* h_flightDistance3dVal = nullptr;
    MonitorElement* h_flightDistance3dSig = nullptr;

    ObjME OnlineTrkEff_Pt;
    ObjME OnlineTrkEff_Eta;
    ObjME OnlineTrkEff_3d_ip_distance;
    ObjME OnlineTrkEff_3d_ip_sig;
    ObjME OnlineTrkFake_Pt;
    ObjME OnlineTrkFake_Eta;
    ObjME OnlineTrkFake_3d_ip_distance;
    ObjME OnlineTrkFake_3d_ip_sig;
    // MonitorElement*  n_pixel_hits_;
    // MonitorElement*  n_total_hits_;

  private:
    int prescaleUsed_;
    std::string pathName_;
    std::string filterName_;
    std::string processName_;
    int objectType_;
    std::string triggerType_;
  };

  class PathInfoCollection : public std::vector<PathInfo> {
  public:
    PathInfoCollection() : std::vector<PathInfo>(){};
    std::vector<PathInfo>::iterator find(const std::string& pathName) { return std::find(begin(), end(), pathName); }
  };

  PathInfoCollection hltPathsAll_;
};

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;

BTVHLTOfflineSource::BTVHLTOfflineSource(const edm::ParameterSet& iConfig)
    : dirname_(iConfig.getUntrackedParameter("dirname", std::string("HLT/BTV/"))),
      processname_(iConfig.getParameter<std::string>("processname")),
      verbose_(iConfig.getUntrackedParameter<bool>("verbose", false)),
      triggerSummaryLabel_(iConfig.getParameter<edm::InputTag>("triggerSummaryLabel")),
      triggerResultsLabel_(iConfig.getParameter<edm::InputTag>("triggerResultsLabel")),
      turnon_threshold_loose_(iConfig.getParameter<double>("turnon_threshold_loose")),
      turnon_threshold_medium_(iConfig.getParameter<double>("turnon_threshold_medium")),
      turnon_threshold_tight_(iConfig.getParameter<double>("turnon_threshold_tight")),
      offlineDiscrTokenb_(consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("offlineDiscrLabelb"))),
      offlineDiscrTokenbb_(
          consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("offlineDiscrLabelbb"))),
      offlineIPToken_(consumes<View<BaseTagInfo>>(iConfig.getParameter<edm::InputTag>("offlineIPLabel"))),

      hltFastPVToken_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("hltFastPVLabel"))),
      hltPFPVToken_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("hltPFPVLabel"))),
      hltCaloPVToken_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("hltCaloPVLabel"))),
      offlinePVToken_(consumes<std::vector<reco::Vertex>>(iConfig.getParameter<edm::InputTag>("offlinePVLabel"))),
      triggerResultsToken(consumes<edm::TriggerResults>(triggerResultsLabel_)),
      triggerResultsFUToken(consumes<edm::TriggerResults>(
          edm::InputTag(triggerResultsLabel_.label(), triggerResultsLabel_.instance(), std::string("FU")))),
      triggerSummaryToken(consumes<trigger::TriggerEvent>(triggerSummaryLabel_)),
      triggerSummaryFUToken(consumes<trigger::TriggerEvent>(
          edm::InputTag(triggerSummaryLabel_.label(), triggerSummaryLabel_.instance(), std::string("FU")))),
      shallowTagInfosTokenCalo_(
          consumes<vector<reco::ShallowTagInfo>>(edm::InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfosCalo"))),
      shallowTagInfosTokenPf_(
          consumes<vector<reco::ShallowTagInfo>>(edm::InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfos"))),
      SVTagInfosTokenCalo_(consumes<std::vector<reco::SecondaryVertexTagInfo>>(
          edm::InputTag("hltInclusiveSecondaryVertexFinderTagInfos"))),
      SVTagInfosTokenPf_(consumes<std::vector<SVTagInfo>>(edm::InputTag("hltDeepSecondaryVertexTagInfosPF"))),
      caloTagsToken_(consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("onlineDiscrLabelCalo"))),
      pfTagsToken_(consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("onlineDiscrLabelPF"))),
      minDecayLength_(iConfig.getParameter<double>("minDecayLength")),
      maxDecayLength_(iConfig.getParameter<double>("maxDecayLength")),
      minJetDistance_(iConfig.getParameter<double>("minJetDistance")),
      maxJetDistance_(iConfig.getParameter<double>("maxJetDistance")),
      dRTrackMatch_(iConfig.getParameter<double>("dRTrackMatch")) {
  std::vector<edm::ParameterSet> paths = iConfig.getParameter<std::vector<edm::ParameterSet>>("pathPairs");
  for (const auto& path : paths) {
    custompathnamepairs_.push_back(
        make_pair(path.getParameter<std::string>("pathname"), path.getParameter<std::string>("pathtype")));
  }
}

BTVHLTOfflineSource::~BTVHLTOfflineSource() = default;

void BTVHLTOfflineSource::dqmBeginRun(const edm::Run& run, const edm::EventSetup& c) {
  bool changed = true;
  if (!hltConfig_.init(run, c, processname_, changed)) {
    LogDebug("BTVHLTOfflineSource") << "HLTConfigProvider failed to initialize.";
  }

  for (unsigned int idx = 0; idx != hltConfig_.size(); ++idx) {
    const auto& pathname = hltConfig_.triggerName(idx);

    for (const auto& custompathnamepair : custompathnamepairs_) {
      if (pathname.find(custompathnamepair.first) != std::string::npos) {
        hltPathsAll_.push_back(PathInfo(1, pathname, "dummy", processname_, 0, custompathnamepair.second));
      }
    }
  }
}

void BTVHLTOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup) {
  edm::Handle<edm::TriggerResults> triggerResults;
  iEvent.getByToken(triggerResultsToken, triggerResults);
  if (!triggerResults.isValid()) {
    iEvent.getByToken(triggerResultsFUToken, triggerResults);
    if (!triggerResults.isValid()) {
      edm::LogInfo("BTVHLTOfflineSource") << "TriggerResults not found, skipping event";
      return;
    }
  }

  const edm::TriggerNames& triggerNames = iEvent.triggerNames(*triggerResults);

  edm::Handle<trigger::TriggerEvent> triggerObj;
  iEvent.getByToken(triggerSummaryToken, triggerObj);
  if (!triggerObj.isValid()) {
    iEvent.getByToken(triggerSummaryFUToken, triggerObj);
    if (!triggerObj.isValid()) {
      edm::LogInfo("BTVHLTOfflineSource") << "TriggerEvent not found, skipping event";
      return;
    }
  }

  edm::Handle<reco::JetTagCollection> caloTags;
  iEvent.getByToken(caloTagsToken_, caloTags);

  edm::Handle<reco::JetTagCollection> pfTags;
  iEvent.getByToken(pfTagsToken_, pfTags);

  Handle<reco::VertexCollection> VertexHandler;

  Handle<reco::JetTagCollection> offlineJetTagHandlerb;
  iEvent.getByToken(offlineDiscrTokenb_, offlineJetTagHandlerb);

  Handle<reco::JetTagCollection> offlineJetTagHandlerbb;
  iEvent.getByToken(offlineDiscrTokenbb_, offlineJetTagHandlerbb);

  Handle<View<BaseTagInfo>> offlineIPTagHandle;
  iEvent.getByToken(offlineIPToken_, offlineIPTagHandle);

  Handle<reco::VertexCollection> offlineVertexHandler;
  iEvent.getByToken(offlinePVToken_, offlineVertexHandler);

  if (verbose_ && iEvent.id().event() % 10000 == 0)
    cout << "Run = " << iEvent.id().run() << ", LS = " << iEvent.luminosityBlock()
         << ", Event = " << iEvent.id().event() << endl;

  if (!triggerResults.isValid())
    return;

  edm::Handle<std::vector<SVTagInfo>> jetSVTagsCollPF;
  edm::Handle<std::vector<reco::SecondaryVertexTagInfo>> jetSVTagsCollCalo;

  for (auto& v : hltPathsAll_) {
    unsigned index = triggerNames.triggerIndex(v.getPath());
    if (!(index < triggerNames.size())) {
      edm::LogInfo("BTVHLTOfflineSource") << "Path " << v.getPath() << " not in menu, skipping event";
      continue;
    }

    if (!triggerResults->accept(index)) {
      edm::LogInfo("BTVHLTOfflineSource") << "Path " << v.getPath() << " not accepted, skipping event";
      continue;
    }

    if (v.getTriggerType() == "PF") {
      iEvent.getByToken(SVTagInfosTokenPf_, jetSVTagsCollPF);
    } else {
      iEvent.getByToken(SVTagInfosTokenCalo_, jetSVTagsCollCalo);
    }

    // PF and Calo btagging
    if ((v.getTriggerType() == "PF" && pfTags.isValid()) ||
        (v.getTriggerType() == "Calo" && caloTags.isValid() && !caloTags->empty())) {
      const auto& iter = (v.getTriggerType() == "PF") ? pfTags->begin() : caloTags->begin();

      float Discr_online = iter->second;
      if (Discr_online < 0)
        Discr_online = -0.05;

      v.Discr->Fill(Discr_online);
      v.Pt->Fill(iter->first->pt());
      v.Eta->Fill(iter->first->eta());

      if (offlineJetTagHandlerb.isValid()) {
        for (auto const& iterOffb : *offlineJetTagHandlerb) {
          float DR = reco::deltaR(iterOffb.first->eta(), iterOffb.first->phi(), iter->first->eta(), iter->first->phi());
          if (DR < 0.3) {
            float Discr_offline = iterOffb.second;

            // offline probb and probbb must be added (if probbb isn't specified, it'll just use probb)
            if (offlineJetTagHandlerbb.isValid()) {
              for (auto const& iterOffbb : *offlineJetTagHandlerbb) {
                DR = reco::deltaR(
                    iterOffbb.first->eta(), iterOffbb.first->phi(), iter->first->eta(), iter->first->phi());
                if (DR < 0.3) {
                  Discr_offline += iterOffbb.second;
                  break;
                }
              }
            }

            if (Discr_offline < 0)
              Discr_offline = -0.05;
            v.Discr_HLTvsRECO->Fill(Discr_online, Discr_offline);
            v.Discr_HLTMinusRECO->Fill(Discr_online - Discr_offline);

            v.Discr_turnon_loose.denominator->Fill(Discr_offline);
            v.Discr_turnon_medium.denominator->Fill(Discr_offline);
            v.Discr_turnon_tight.denominator->Fill(Discr_offline);

            if (Discr_online > turnon_threshold_loose_)
              v.Discr_turnon_loose.numerator->Fill(Discr_offline);
            if (Discr_online > turnon_threshold_medium_)
              v.Discr_turnon_medium.numerator->Fill(Discr_offline);
            if (Discr_online > turnon_threshold_tight_)
              v.Discr_turnon_tight.numerator->Fill(Discr_offline);

            break;
          }
        }
      }  ///offline

      bool pfSVTagCollValid = (v.getTriggerType() == "PF" && jetSVTagsCollPF.isValid());
      bool caloSVTagCollValid = (v.getTriggerType() == "Calo" && jetSVTagsCollCalo.isValid());
      if (offlineIPTagHandle.isValid() && (pfSVTagCollValid || caloSVTagCollValid)) {
        std::vector<float> offlineIP3D;
        std::vector<float> offlineIP3DSig;
        std::vector<const reco::Track*> offlineTracks = getOfflineBTagTracks(
            iter->first->eta(), iter->first->phi(), offlineIPTagHandle, offlineIP3D, offlineIP3DSig);
        std::vector<const reco::Track*> onlineTracks;
        std::vector<float> onlineIP3D;
        std::vector<float> onlineIP3DSig;
        if (pfSVTagCollValid)
          onlineTracks = getOnlineBTagTracks<SVTagInfo>(
              iter->first->eta(), iter->first->phi(), jetSVTagsCollPF, onlineIP3D, onlineIP3DSig);
        if (caloSVTagCollValid)
          onlineTracks = getOnlineBTagTracks<reco::SecondaryVertexTagInfo>(
              iter->first->eta(), iter->first->phi(), jetSVTagsCollCalo, onlineIP3D, onlineIP3DSig);

        for (unsigned int iOffTrk = 0; iOffTrk < offlineTracks.size(); ++iOffTrk) {
          const reco::Track* offTrk = offlineTracks.at(iOffTrk);
          bool hasMatch = false;
          float offTrkEta = offTrk->eta();
          float offTrkPhi = offTrk->phi();

          for (const reco::Track* onTrk : onlineTracks) {
            float DR = reco::deltaR(offTrkEta, offTrkPhi, onTrk->eta(), onTrk->phi());
            if (DR < dRTrackMatch_) {
              hasMatch = true;
            }
          }

          float offTrkPt = offTrk->pt();
          v.OnlineTrkEff_Pt.denominator->Fill(offTrkPt);
          if (hasMatch)
            v.OnlineTrkEff_Pt.numerator->Fill(offTrkPt);

          v.OnlineTrkEff_Eta.denominator->Fill(offTrkEta);
          if (hasMatch)
            v.OnlineTrkEff_Eta.numerator->Fill(offTrkEta);

          v.OnlineTrkEff_3d_ip_distance.denominator->Fill(offlineIP3D.at(iOffTrk));
          if (hasMatch)
            v.OnlineTrkEff_3d_ip_distance.numerator->Fill(offlineIP3D.at(iOffTrk));

          v.OnlineTrkEff_3d_ip_sig.denominator->Fill(offlineIP3DSig.at(iOffTrk));
          if (hasMatch)
            v.OnlineTrkEff_3d_ip_sig.numerator->Fill(offlineIP3DSig.at(iOffTrk));
        }

        for (unsigned int iOnTrk = 0; iOnTrk < onlineTracks.size(); ++iOnTrk) {
          const reco::Track* onTrk = onlineTracks.at(iOnTrk);
          bool hasMatch = false;
          float onTrkEta = onTrk->eta();
          float onTrkPhi = onTrk->phi();

          for (const reco::Track* offTrk : offlineTracks) {
            float DR = reco::deltaR(onTrkEta, onTrkPhi, offTrk->eta(), offTrk->phi());
            if (DR < dRTrackMatch_) {
              hasMatch = true;
            }
          }

          float onTrkPt = onTrk->pt();
          v.OnlineTrkFake_Pt.denominator->Fill(onTrkPt);
          if (!hasMatch)
            v.OnlineTrkFake_Pt.numerator->Fill(onTrkPt);

          v.OnlineTrkFake_Eta.denominator->Fill(onTrkEta);
          if (!hasMatch)
            v.OnlineTrkFake_Eta.numerator->Fill(onTrkEta);

          v.OnlineTrkFake_3d_ip_distance.denominator->Fill(onlineIP3D.at(iOnTrk));
          if (!hasMatch)
            v.OnlineTrkFake_3d_ip_distance.numerator->Fill(onlineIP3D.at(iOnTrk));

          v.OnlineTrkFake_3d_ip_sig.denominator->Fill(onlineIP3DSig.at(iOnTrk));
          if (!hasMatch)
            v.OnlineTrkFake_3d_ip_sig.numerator->Fill(onlineIP3DSig.at(iOnTrk));
        }
      }

      if (v.getTriggerType() == "PF") {
        iEvent.getByToken(hltPFPVToken_, VertexHandler);
      } else {
        iEvent.getByToken(hltFastPVToken_, VertexHandler);
      }
      if (VertexHandler.isValid()) {
        v.PVz->Fill(VertexHandler->begin()->z());
        if (offlineVertexHandler.isValid()) {
          v.PVz_HLTMinusRECO->Fill(VertexHandler->begin()->z() - offlineVertexHandler->begin()->z());
        }
      }
    }  // caloTagsValid or PFTagsValid

    // specific to Calo b-tagging
    if (caloTags.isValid() && v.getTriggerType() == "Calo" && !caloTags->empty()) {
      iEvent.getByToken(hltCaloPVToken_, VertexHandler);
      if (VertexHandler.isValid()) {
        v.fastPVz->Fill(VertexHandler->begin()->z());
        if (offlineVertexHandler.isValid()) {
          v.fastPVz_HLTMinusRECO->Fill(VertexHandler->begin()->z() - offlineVertexHandler->begin()->z());
        }
      }
    }

    // additional plots from tag info collections
    /////////////////////////////////////////////

    edm::Handle<std::vector<reco::ShallowTagInfo>> shallowTagInfosCalo;
    iEvent.getByToken(shallowTagInfosTokenCalo_, shallowTagInfosCalo);

    edm::Handle<std::vector<reco::ShallowTagInfo>> shallowTagInfosPf;
    iEvent.getByToken(shallowTagInfosTokenPf_, shallowTagInfosPf);

    //    edm::Handle<std::vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<edm::RefVector<std::vector<reco::Track>, reco::Track, edm::refhelper::FindUsingAdvance<std::vector<reco::Track>, reco::Track> >, reco::JTATagInfo>, reco::Vertex> > > caloTagInfos;
    //    iEvent.getByToken(caloTagInfosToken_, caloTagInfos);

    //    edm::Handle<std::vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<edm::RefVector<std::vector<reco::Track>, reco::Track, edm::refhelper::FindUsingAdvance<std::vector<reco::Track>, reco::Track> >, reco::JTATagInfo>, reco::Vertex> > > pfTagInfos;
    //    iEvent.getByToken(pfTagInfosToken_, pfTagInfos);

    // first try to get info from shallowTagInfos ...
    if ((v.getTriggerType() == "PF" && shallowTagInfosPf.isValid()) ||
        (v.getTriggerType() == "Calo" && shallowTagInfosCalo.isValid())) {
      const auto& shallowTagInfoCollection = (v.getTriggerType() == "PF") ? shallowTagInfosPf : shallowTagInfosCalo;
      for (const auto& shallowTagInfo : *shallowTagInfoCollection) {
        const auto& tagVars = shallowTagInfo.taggingVariables();

        // n secondary vertices and n selected tracks
        for (const auto& tagVar : tagVars.getList(reco::btau::jetNSecondaryVertices, false)) {
          v.h_jetNSecondaryVertices->Fill(tagVar);
          v.n_vtx->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::jetNSelectedTracks, false)) {
          v.n_sel_tracks->Fill(tagVar);
          v.h_jetNSelectedTracks->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::jetPt, false)) {
          v.h_jet_pt->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::jetEta, false)) {
          v.h_jet_eta->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSumJetEtRatio, false)) {
          v.h_trackSumJetEtRatio->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSumJetDeltaR, false)) {
          v.h_trackSumJetDeltaR->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::vertexCategory, false)) {
          v.h_vertexCategory->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSip2dValAboveCharm, false)) {
          v.h_trackSip2dValAboveCharm->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSip2dSigAboveCharm, false)) {
          v.h_trackSip2dSigAboveCharm->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSip3dValAboveCharm, false)) {
          v.h_trackSip3dValAboveCharm->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::trackSip3dSigAboveCharm, false)) {
          v.h_trackSip3dSigAboveCharm->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::jetNTracksEtaRel, false)) {
          v.h_jetNTracksEtaRel->Fill(tagVar);
        }

        // impact parameter
        // and new info
        const auto& trackSip3dVal = tagVars.getList(reco::btau::trackSip3dVal, false);
        const auto& trackSip3dSig = tagVars.getList(reco::btau::trackSip3dSig, false);
        const auto& trackJetDistVal = tagVars.getList(reco::btau::trackJetDistVal, false);
        const auto& trackPtRel = tagVars.getList(reco::btau::trackPtRel, false);
        const auto& trackSip2dSig = tagVars.getList(reco::btau::trackSip2dSig, false);
        const auto& trackDeltaR = tagVars.getList(reco::btau::trackDeltaR, false);
        const auto& trackPtRatio = tagVars.getList(reco::btau::trackPtRatio, false);
        const auto& trackDecayLenVal = tagVars.getList(reco::btau::trackDecayLenVal, false);
        const auto& trackEtaRel = tagVars.getList(reco::btau::trackEtaRel, false);

        for (unsigned i_trk = 0; i_trk < trackEtaRel.size(); i_trk++) {
          v.h_trackEtaRel->Fill(trackEtaRel[i_trk]);
        }

        for (unsigned i_trk = 0; i_trk < trackJetDistVal.size(); i_trk++) {
          v.h_trackJetDistVal->Fill(trackJetDistVal[i_trk]);
        }

        for (unsigned i_trk = 0; i_trk < trackPtRel.size(); i_trk++) {
          v.h_trackPtRel->Fill(trackPtRel[i_trk]);
        }

        for (unsigned i_trk = 0; i_trk < trackDeltaR.size(); i_trk++) {
          v.h_trackDeltaR->Fill(trackDeltaR[i_trk]);
        }

        for (unsigned i_trk = 0; i_trk < trackPtRatio.size(); i_trk++) {
          v.h_trackPtRatio->Fill(trackPtRatio[i_trk]);
        }

        for (unsigned i_trk = 0; i_trk < trackDecayLenVal.size(); i_trk++) {
          v.h_trackDecayLenVal->Fill(trackDecayLenVal[i_trk]);
        }

        for (unsigned i_trk = 0; i_trk < trackSip3dVal.size(); i_trk++) {
          float val = trackSip3dVal[i_trk];
          float sig = trackSip3dSig[i_trk];
          v.h_3d_ip_distance->Fill(val);
          v.h_3d_ip_error->Fill(val / sig);
          v.h_3d_ip_sig->Fill(sig);

          v.h_trackSip2dSig->Fill(trackSip2dSig[i_trk]);
        }

        // vertex mass and tracks per vertex
        for (const auto& tagVar : tagVars.getList(reco::btau::vertexMass, false)) {
          v.vtx_mass->Fill(tagVar);
        }
        for (const auto& tagVar : tagVars.getList(reco::btau::vertexNTracks, false)) {
          v.n_vtx_trks->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::vertexEnergyRatio, false)) {
          v.h_vertexEnergyRatio->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::vertexJetDeltaR, false)) {
          v.h_vertexJetDeltaR->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::flightDistance2dVal, false)) {
          v.h_flightDistance2dVal->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::flightDistance2dSig, false)) {
          v.h_flightDistance2dSig->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::flightDistance3dVal, false)) {
          v.h_flightDistance3dVal->Fill(tagVar);
        }

        for (const auto& tagVar : tagVars.getList(reco::btau::flightDistance3dSig, false)) {
          v.h_flightDistance3dSig->Fill(tagVar);
        }

        // // track N total/pixel hits
        // for (const auto & tagVar : tagVars.getList(reco::btau::trackNPixelHits, false)) {
        //   v.n_pixel_hits->Fill(tagVar);}
        // for (const auto & tagVar : tagVars.getList(reco::btau::trackNTotalHits, false)) {
        //   v.n_total_hits->Fill(tagVar);}
      }
    }

    // ... otherwise from usual tag infos.
    // else
    // if (   (v.getTriggerType() == "PF"   && pfTagInfos.isValid())
    //     || (v.getTriggerType() == "Calo" && caloTagInfos.isValid()) )
    // {
    //   const auto & DiscrTagInfoCollection = (v.getTriggerType() == "PF") ? pfTagInfos : caloTagInfos;

    //   // loop over secondary vertex tag infos
    //   for (const auto & DiscrTagInfo : *DiscrTagInfoCollection) {
    //     v.n_vtx->Fill(DiscrTagInfo.nVertexCandidates());
    //     v.n_sel_tracks->Fill(DiscrTagInfo.nSelectedTracks());

    //     // loop over selected tracks in each tag info
    //     for (unsigned i_trk=0; i_trk < DiscrTagInfo.nSelectedTracks(); i_trk++) {
    //       const auto & ip3d = DiscrTagInfo.trackIPData(i_trk).ip3d;
    //       v.h_3d_ip_distance->Fill(ip3d.value());
    //       v.h_3d_ip_error->Fill(ip3d.error());
    //       v.h_3d_ip_sig->Fill(ip3d.significance());
    //     }

    //     // loop over vertex candidates in each tag info
    //     for (unsigned i_sv=0; i_sv < DiscrTagInfo.nVertexCandidates(); i_sv++) {
    //       const auto & sv = DiscrTagInfo.secondaryVertex(i_sv);
    //       v.vtx_mass->Fill(sv.p4().mass());
    //       v.n_vtx_trks->Fill(sv.nTracks());

    //       // loop over tracks for number of pixel and total hits
    //       const auto & trkIPTagInfo = DiscrTagInfo.trackIPTagInfoRef().get();
    //       for (const auto & trk : trkIPTagInfo->selectedTracks()) {
    //         v.n_pixel_hits->Fill(trk.get()->hitPattern().numberOfValidPixelHits());
    //         v.n_total_hits->Fill(trk.get()->hitPattern().numberOfValidHits());
    //       }
    //     }
    //   }
    // }
  }  //end paths loop
}

std::vector<const reco::Track*> BTVHLTOfflineSource::getOfflineBTagTracks(float hltJetEta,
                                                                          float hltJetPhi,
                                                                          Handle<View<BaseTagInfo>> offlineIPTagHandle,
                                                                          std::vector<float>& offlineIP3D,
                                                                          std::vector<float>& offlineIP3DSig) {
  std::vector<const reco::Track*> offlineTracks;

  for (auto const& iterOffIP : *offlineIPTagHandle) {
    float DR = reco::deltaR(iterOffIP.jet()->eta(), iterOffIP.jet()->phi(), hltJetEta, hltJetPhi);

    if (DR > 0.3)
      continue;

    const reco::IPTagInfo<vector<reco::CandidatePtr>, reco::JetTagInfo>* tagInfo =
        dynamic_cast<const reco::IPTagInfo<vector<reco::CandidatePtr>, reco::JetTagInfo>*>(&iterOffIP);

    if (!tagInfo) {
      throw cms::Exception("Configuration")
          << "BTagPerformanceAnalyzer: Extended TagInfo not of type TrackIPTagInfo. " << std::endl;
    }

    const GlobalPoint pv(tagInfo->primaryVertex()->position().x(),
                         tagInfo->primaryVertex()->position().y(),
                         tagInfo->primaryVertex()->position().z());

    const std::vector<reco::btag::TrackIPData>& ip = tagInfo->impactParameterData();

    std::vector<std::size_t> sortedIndices = tagInfo->sortedIndexes(reco::btag::IP2DSig);
    std::vector<reco::CandidatePtr> sortedTracks = tagInfo->sortedTracks(sortedIndices);
    std::vector<std::size_t> selectedIndices;
    vector<reco::CandidatePtr> selectedTracks;
    for (unsigned int n = 0; n != sortedIndices.size(); ++n) {
      double decayLength = (ip[sortedIndices[n]].closestToJetAxis - pv).mag();
      double jetDistance = ip[sortedIndices[n]].distanceToJetAxis.value();
      if (decayLength > minDecayLength_ && decayLength < maxDecayLength_ && fabs(jetDistance) >= minJetDistance_ &&
          fabs(jetDistance) < maxJetDistance_) {
        selectedIndices.push_back(sortedIndices[n]);
        selectedTracks.push_back(sortedTracks[n]);
      }
    }

    for (unsigned int n = 0; n != selectedIndices.size(); ++n) {
      const reco::Track* track = reco::btag::toTrack(selectedTracks[n]);
      offlineTracks.push_back(track);
      offlineIP3D.push_back(ip[n].ip3d.value());
      offlineIP3DSig.push_back(ip[n].ip3d.significance());
    }
  }
  return offlineTracks;
}

template <class Base>
std::vector<const reco::Track*> BTVHLTOfflineSource::getOnlineBTagTracks(float hltJetEta,
                                                                         float hltJetPhi,
                                                                         edm::Handle<std::vector<Base>> jetSVTagsColl,
                                                                         std::vector<float>& onlineIP3D,
                                                                         std::vector<float>& onlineIP3DSig) {
  std::vector<const reco::Track*> onlineTracks;

  for (auto iterTI = jetSVTagsColl->begin(); iterTI != jetSVTagsColl->end(); ++iterTI) {
    float DR = reco::deltaR(iterTI->jet()->eta(), iterTI->jet()->phi(), hltJetEta, hltJetPhi);
    if (DR > 0.3)
      continue;

    const auto& ipInfo = *(iterTI->trackIPTagInfoRef().get());
    const std::vector<reco::btag::TrackIPData>& ip = ipInfo.impactParameterData();

    unsigned int trackSize = ipInfo.selectedTracks().size();
    for (unsigned int itt = 0; itt < trackSize; ++itt) {
      const auto ptrackRef = (ipInfo.selectedTracks()[itt]);

      if (ptrackRef.isAvailable()) {
        const reco::Track* ptrackPtr = reco::btag::toTrack(ptrackRef);
        onlineTracks.push_back(ptrackPtr);
        onlineIP3D.push_back(ip[itt].ip3d.value());
        onlineIP3DSig.push_back(ip[itt].ip3d.significance());
      }
    }
  }
  return onlineTracks;
}

void BTVHLTOfflineSource::bookHistograms(DQMStore::IBooker& iBooker, edm::Run const& run, edm::EventSetup const& c) {
  iBooker.setCurrentFolder(dirname_);
  for (auto& v : hltPathsAll_) {
    std::string trgPathName = HLTConfigProvider::removeVersion(v.getPath());
    std::string subdirName = dirname_ + "/" + trgPathName + v.getTriggerType();
    std::string trigPath = "(" + trgPathName + ")";
    iBooker.setCurrentFolder(subdirName);

    std::string labelname("HLT");
    std::string histoname(labelname + "");
    std::string title(labelname + "");

    histoname = labelname + "_Discr";
    title = labelname + "_Discr " + trigPath;
    v.Discr = iBooker.book1D(histoname.c_str(), title.c_str(), 110, -0.1, 1);

    histoname = labelname + "_Pt";
    title = labelname + "_Pt " + trigPath;
    v.Pt = iBooker.book1D(histoname.c_str(), title.c_str(), 100, 0, 400);

    histoname = labelname + "_Eta";
    title = labelname + "_Eta " + trigPath;
    v.Eta = iBooker.book1D(histoname.c_str(), title.c_str(), 60, -3.0, 3.0);

    histoname = "HLTvsRECO_Discr";
    title = "online discr vs offline discr " + trigPath;
    v.Discr_HLTvsRECO = iBooker.book2D(histoname.c_str(), title.c_str(), 110, -0.1, 1, 110, -0.1, 1);

    histoname = "HLTMinusRECO_Discr";
    title = "online discr minus offline discr " + trigPath;
    v.Discr_HLTMinusRECO = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -1, 1);

    histoname = "Turnon_loose_Discr";
    title = "turn-on with loose threshold " + trigPath;
    v.bookME(iBooker, v.Discr_turnon_loose, histoname, title, 22, -0.1, 1.);

    histoname = "Turnon_medium_Discr";
    title = "turn-on with medium threshold " + trigPath;
    v.bookME(iBooker, v.Discr_turnon_medium, histoname, title, 22, -0.1, 1.);

    histoname = "Turnon_tight_Discr";
    title = "turn-on with tight threshold " + trigPath;
    v.bookME(iBooker, v.Discr_turnon_tight, histoname, title, 22, -0.1, 1.);

    histoname = labelname + "_PVz";
    title = "online z(PV) " + trigPath;
    v.PVz = iBooker.book1D(histoname.c_str(), title.c_str(), 80, -20, 20);

    histoname = labelname + "_fastPVz";
    title = "online z(fastPV) " + trigPath;
    v.fastPVz = iBooker.book1D(histoname.c_str(), title.c_str(), 80, -20, 20);

    histoname = "HLTMinusRECO_PVz";
    title = "online z(PV) - offline z(PV) " + trigPath;
    v.PVz_HLTMinusRECO = iBooker.book1D(histoname.c_str(), title.c_str(), 200, -0.5, 0.5);

    histoname = "HLTMinusRECO_fastPVz";
    title = "online z(fastPV) - offline z(PV) " + trigPath;
    v.fastPVz_HLTMinusRECO = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -2, 2);

    histoname = "n_vtx";
    title = "N vertex candidates " + trigPath;
    v.n_vtx = iBooker.book1D(histoname.c_str(), title.c_str(), 10, -0.5, 9.5);

    histoname = "vtx_mass";
    title = "secondary vertex mass (GeV)" + trigPath;
    v.vtx_mass = iBooker.book1D(histoname.c_str(), title.c_str(), 20, 0, 10);

    histoname = "n_vtx_trks";
    title = "N tracks associated to secondary vertex" + trigPath;
    v.n_vtx_trks = iBooker.book1D(histoname.c_str(), title.c_str(), 20, -0.5, 19.5);

    histoname = "n_sel_tracks";
    title = "N selected tracks" + trigPath;
    v.n_sel_tracks = iBooker.book1D(histoname.c_str(), title.c_str(), 25, -0.5, 24.5);

    histoname = "3d_ip_distance";
    title = "3D IP distance of tracks (cm)" + trigPath;
    v.h_3d_ip_distance = iBooker.book1D(histoname.c_str(), title.c_str(), 40, -0.1, 0.1);

    histoname = "3d_ip_error";
    title = "3D IP error of tracks (cm)" + trigPath;
    v.h_3d_ip_error = iBooker.book1D(histoname.c_str(), title.c_str(), 40, 0., 0.1);

    histoname = "3d_ip_sig";
    title = "3D IP significance of tracks (cm)" + trigPath;
    v.h_3d_ip_sig = iBooker.book1D(histoname.c_str(), title.c_str(), 40, -40, 40);

    //new
    histoname = "jetNSecondaryVertices";
    title = "jet N Secondary Vertices" + trigPath;
    v.h_jetNSecondaryVertices = iBooker.book1D(histoname.c_str(), title.c_str(), 10, -0.5, 9.5);

    histoname = "jet_pt";
    title = "jet pt" + trigPath;
    v.h_jet_pt = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.1, 100);

    histoname = "jet_eta";
    title = "jet eta" + trigPath;
    v.h_jet_eta = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -2.5, 2.5);

    histoname = "trackSumJetEtRatio";
    title = "trackSumJetEtRatio" + trigPath;
    v.h_trackSumJetEtRatio = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.1, 1.5);

    histoname = "trackSip2dValAboveCharm";
    title = "trackSip2dSigAboveCharm" + trigPath;
    v.h_trackSip2dSigAboveCharm = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.2, 0.2);

    histoname = "trackSip2dSigAboveCharm";
    title = "trackSip2dSigAboveCharm" + trigPath;
    v.h_trackSip2dValAboveCharm = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -50, 50);

    histoname = "trackSip3dValAboveCharm";
    title = "trackSip3dValAboveCharm" + trigPath;
    v.h_trackSip3dValAboveCharm = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.2, 0.2);

    histoname = "trackSip3dSigAboveCharm";
    title = "trackSip3dSigAboveCharm" + trigPath;
    v.h_trackSip3dSigAboveCharm = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -50, 50);

    histoname = "jetNSelectedTracks";
    title = "jet N Selected Tracks" + trigPath;
    v.h_jetNSelectedTracks = iBooker.book1D(histoname.c_str(), title.c_str(), 42, -1.5, 40.5);

    histoname = "jetNTracksEtaRel";
    title = "jetNTracksEtaRel" + trigPath;
    v.h_jetNTracksEtaRel = iBooker.book1D(histoname.c_str(), title.c_str(), 42, -1.5, 40.5);

    histoname = "vertexCategory";
    title = "vertex category" + trigPath;
    v.h_vertexCategory = iBooker.book1D(histoname.c_str(), title.c_str(), 4, -1.5, 2.5);

    histoname = "trackSumJetDeltaR";
    title = "trackSumJetDeltaR" + trigPath;
    v.h_trackSumJetDeltaR = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.1, 0.35);

    //new 2 below
    histoname = "trackJetDistVal";
    title = "trackJetDistVal" + trigPath;
    v.h_trackJetDistVal = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -1, 0.01);

    histoname = "trackPtRel";
    title = "track pt rel" + trigPath;
    v.h_trackPtRel = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.1, 7);

    histoname = "trackDeltaR";
    title = "trackDeltaR" + trigPath;
    v.h_trackDeltaR = iBooker.book1D(histoname.c_str(), title.c_str(), 160, -0.05, .47);

    histoname = "trackPtRatio";
    title = "trackPtRatio" + trigPath;
    v.h_trackPtRatio = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.01, 0.3);

    histoname = "trackSip2dSig";
    title = "trackSip2dSig" + trigPath;
    v.h_trackSip2dSig = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -55, 55);

    histoname = "trackDecayLenVal";
    title = "trackDecayLenVal" + trigPath;
    v.h_trackDecayLenVal = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.1, 22);

    histoname = "trackEtaRel";
    title = "trackEtaRel" + trigPath;
    v.h_trackEtaRel = iBooker.book1D(histoname.c_str(), title.c_str(), 31, 0, 30);

    //new 3 below
    histoname = "vertexEnergyRatio";
    title = "vertexEnergyRatio" + trigPath;
    v.h_vertexEnergyRatio = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.1, 3);

    histoname = "vertexJetDeltaR";
    title = "vertexJetDeltaR" + trigPath;
    v.h_vertexJetDeltaR = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.01, 0.4);

    histoname = "flightDistance2dVal";
    title = "flightDistance2dVal" + trigPath;
    v.h_flightDistance2dVal = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.1, 5);

    histoname = "flightDistance2dSig";
    title = "flightDistance2dSig" + trigPath;
    v.h_flightDistance2dSig = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -10, 150);

    histoname = "flightDistance3dVal";
    title = "flightDistance3dVal" + trigPath;
    v.h_flightDistance3dVal = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -0.1, 5);

    histoname = "flightDistance3dSig";
    title = "flightDistance3dSig" + trigPath;
    v.h_flightDistance3dSig = iBooker.book1D(histoname.c_str(), title.c_str(), 100, -10, 150);

    //end new

    histoname = "OnlineTrkEff_Pt";
    title = "Relative Online Trk Efficiency vs Pt " + trigPath;
    v.bookME(iBooker, v.OnlineTrkEff_Pt, histoname, title, 50, -0.5, 20.);

    histoname = "OnlineTrkEff_Eta";
    title = "Relative Online Trk Efficiency vs Eta " + trigPath;
    v.bookME(iBooker, v.OnlineTrkEff_Eta, histoname, title, 60, -3.0, 3.0);

    histoname = "OnlineTrkEff_3d_ip_distance";
    title = "Relative Online Trk Efficiency vs IP3D " + trigPath;
    v.bookME(iBooker, v.OnlineTrkEff_3d_ip_distance, histoname, title, 40, -0.1, 0.1);

    histoname = "OnlineTrkEff_3d_ip_sig";
    title = "Relative Online Trk Efficiency vs IP3D significance " + trigPath;
    v.bookME(iBooker, v.OnlineTrkEff_3d_ip_sig, histoname, title, 40, -40, 40);

    histoname = "OnlineTrkFake_Pt";
    title = "Relative Online Trk Fake Rate  vs Pt " + trigPath;
    v.bookME(iBooker, v.OnlineTrkFake_Pt, histoname, title, 50, -0.5, 20.);

    histoname = "OnlineTrkFake_Eta";
    title = "Relative Online Trk Fake Rate vs Eta " + trigPath;
    v.bookME(iBooker, v.OnlineTrkFake_Eta, histoname, title, 60, -3.0, 3.0);

    histoname = "OnlineTrkFake_3d_ip_distance";
    title = "Relative Online Trk Fake Rate vs IP3D " + trigPath;
    v.bookME(iBooker, v.OnlineTrkFake_3d_ip_distance, histoname, title, 40, -0.1, 0.1);

    histoname = "OnlineTrkFake_3d_ip_sig";
    title = "Relative Online Trk Fake Rate vs IP3D significance " + trigPath;
    v.bookME(iBooker, v.OnlineTrkFake_3d_ip_sig, histoname, title, 40, -40, 40);

    // histoname = "n_pixel_hits";
    // title = "N pixel hits"+trigPath;
    // v.n_pixel_hits = iBooker.book1D(histoname.c_str(), title.c_str(), 16, -0.5, 15.5);

    // histoname = "n_total_hits";
    // title = "N hits"+trigPath;
    // v.n_total_hits = iBooker.book1D(histoname.c_str(), title.c_str(), 40, -0.5, 39.5);
  }
}

// Define this as a plug-in
DEFINE_FWK_MODULE(BTVHLTOfflineSource);
