/*
  BTVHLTOffline DQM code
*/
//
// Originally created by:  Anne-Catherine Le Bihan
//                         June 2015
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

#include "TMath.h"
#include "TPRegexp.h"

class BTVHLTOfflineSource : public DQMEDAnalyzer {
public:
  explicit BTVHLTOfflineSource(const edm::ParameterSet&);
  ~BTVHLTOfflineSource() override;

private:
  void analyze(const edm::Event&, const edm::EventSetup&) override;
  void bookHistograms(DQMStore::IBooker&, edm::Run const& run, edm::EventSetup const& c) override;
  void dqmBeginRun(edm::Run const& run, edm::EventSetup const& c) override;

  std::string dirname_;
  std::string processname_;
  bool verbose_;

  std::vector<std::pair<std::string, std::string> > custompathnamepairs_;

  edm::InputTag triggerSummaryLabel_;
  edm::InputTag triggerResultsLabel_;

  float turnon_threshold_loose_;
  float turnon_threshold_medium_;
  float turnon_threshold_tight_;

  edm::EDGetTokenT<reco::JetTagCollection> offlineDiscrTokenb_;
  edm::EDGetTokenT<reco::JetTagCollection> offlineDiscrTokenbb_;

  edm::EDGetTokenT<std::vector<reco::Vertex> > hltFastPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > hltPFPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > hltCaloPVToken_;
  edm::EDGetTokenT<std::vector<reco::Vertex> > offlinePVToken_;

  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken;
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsFUToken;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerSummaryToken;
  edm::EDGetTokenT<trigger::TriggerEvent> triggerSummaryFUToken;

  edm::EDGetTokenT<std::vector<reco::ShallowTagInfo> > shallowTagInfosTokenCalo_;
  edm::EDGetTokenT<std::vector<reco::ShallowTagInfo> > shallowTagInfosTokenPf_;

  edm::EDGetTokenT<reco::JetTagCollection> caloTagsToken_;
  edm::EDGetTokenT<reco::JetTagCollection> pfTagsToken_;
  edm::Handle<reco::JetTagCollection> caloTags;
  edm::Handle<reco::JetTagCollection> pfTags;

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
      hltFastPVToken_(consumes<std::vector<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("hltFastPVLabel"))),
      hltPFPVToken_(consumes<std::vector<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("hltPFPVLabel"))),
      hltCaloPVToken_(consumes<std::vector<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("hltCaloPVLabel"))),
      offlinePVToken_(consumes<std::vector<reco::Vertex> >(iConfig.getParameter<edm::InputTag>("offlinePVLabel"))),
      triggerResultsToken(consumes<edm::TriggerResults>(triggerResultsLabel_)),
      triggerResultsFUToken(consumes<edm::TriggerResults>(
          edm::InputTag(triggerResultsLabel_.label(), triggerResultsLabel_.instance(), std::string("FU")))),
      triggerSummaryToken(consumes<trigger::TriggerEvent>(triggerSummaryLabel_)),
      triggerSummaryFUToken(consumes<trigger::TriggerEvent>(
          edm::InputTag(triggerSummaryLabel_.label(), triggerSummaryLabel_.instance(), std::string("FU")))),
      shallowTagInfosTokenCalo_(
          consumes<vector<reco::ShallowTagInfo> >(edm::InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfosCalo"))),
      shallowTagInfosTokenPf_(
          consumes<vector<reco::ShallowTagInfo> >(edm::InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfos"))),
      caloTagsToken_(consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("onlineDiscrLabelCalo"))),
      pfTagsToken_(consumes<reco::JetTagCollection>(iConfig.getParameter<edm::InputTag>("onlineDiscrLabelPF"))) {
  std::vector<edm::ParameterSet> paths = iConfig.getParameter<std::vector<edm::ParameterSet> >("pathPairs");
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

  Handle<reco::VertexCollection> offlineVertexHandler;
  iEvent.getByToken(offlinePVToken_, offlineVertexHandler);

  if (verbose_ && iEvent.id().event() % 10000 == 0)
    cout << "Run = " << iEvent.id().run() << ", LS = " << iEvent.luminosityBlock()
         << ", Event = " << iEvent.id().event() << endl;

  if (!triggerResults.isValid())
    return;

  for (auto& v : hltPathsAll_) {
    unsigned index = triggerNames.triggerIndex(v.getPath());
    if (!(index < triggerNames.size())) {
      continue;
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
    }

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

    edm::Handle<std::vector<reco::ShallowTagInfo> > shallowTagInfosCalo;
    iEvent.getByToken(shallowTagInfosTokenCalo_, shallowTagInfosCalo);

    edm::Handle<std::vector<reco::ShallowTagInfo> > shallowTagInfosPf;
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
          v.n_vtx->Fill(tagVar);
        }
        for (const auto& tagVar : tagVars.getList(reco::btau::jetNSelectedTracks, false)) {
          v.n_sel_tracks->Fill(tagVar);
        }

        // impact parameter
        const auto& trackSip3dVal = tagVars.getList(reco::btau::trackSip3dVal, false);
        const auto& trackSip3dSig = tagVars.getList(reco::btau::trackSip3dSig, false);
        for (unsigned i_trk = 0; i_trk < trackSip3dVal.size(); i_trk++) {
          float val = trackSip3dVal[i_trk];
          float sig = trackSip3dSig[i_trk];
          v.h_3d_ip_distance->Fill(val);
          v.h_3d_ip_error->Fill(val / sig);
          v.h_3d_ip_sig->Fill(sig);
        }

        // vertex mass and tracks per vertex
        for (const auto& tagVar : tagVars.getList(reco::btau::vertexMass, false)) {
          v.vtx_mass->Fill(tagVar);
        }
        for (const auto& tagVar : tagVars.getList(reco::btau::vertexNTracks, false)) {
          v.n_vtx_trks->Fill(tagVar);
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
  }
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
