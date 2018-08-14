#include "DQMOffline/Trigger/plugins/BTVHLTOfflineSource.h"

#include "FWCore/Common/interface/TriggerNames.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

#include "DQMServices/Core/interface/MonitorElement.h"

#include "CommonTools/UtilAlgos/interface/DeltaR.h"
#include "DataFormats/VertexReco/interface/Vertex.h"

#include "TMath.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TProfile.h"
#include "TPRegexp.h"

#include <cmath>
#include <iostream>

using namespace edm;
using namespace reco;
using namespace std;
using namespace trigger;

BTVHLTOfflineSource::BTVHLTOfflineSource(const edm::ParameterSet& iConfig)
{
  LogDebug("BTVHLTOfflineSource") << "constructor....";

  dirname_                  = iConfig.getUntrackedParameter("dirname",std::string("HLT/BTV/"));
  processname_              = iConfig.getParameter<std::string>("processname");
  verbose_                  = iConfig.getUntrackedParameter< bool >("verbose", false);
  triggerSummaryLabel_      = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_      = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  turnon_threshold_loose_   = iConfig.getParameter<double>("turnon_threshold_loose");
  turnon_threshold_medium_  = iConfig.getParameter<double>("turnon_threshold_medium");
  turnon_threshold_tight_   = iConfig.getParameter<double>("turnon_threshold_tight");
  triggerSummaryToken       = consumes <trigger::TriggerEvent> (triggerSummaryLabel_);
  triggerResultsToken       = consumes <edm::TriggerResults>   (triggerResultsLabel_);
  triggerSummaryFUToken     = consumes <trigger::TriggerEvent> (edm::InputTag(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(),std::string("FU")));
  triggerResultsFUToken     = consumes <edm::TriggerResults>   (edm::InputTag(triggerResultsLabel_.label(),triggerResultsLabel_.instance(),std::string("FU")));
  shallowTagInfosTokenCalo_ = consumes<vector<reco::ShallowTagInfo> > (edm::InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfosCalo"));
  shallowTagInfosTokenPf_   = consumes<vector<reco::ShallowTagInfo> > (edm::InputTag("hltDeepCombinedSecondaryVertexBJetTagsInfos"));
  // caloTagInfosToken_       = consumes<vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<edm::RefVector<vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<vector<reco::Track>,reco::Track> >,reco::JTATagInfo>,reco::Vertex> > > (
  //                               edm::InputTag("hltCombinedSecondaryVertexBJetTagsCalo"));
  // pfTagInfosToken_         = consumes<vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<edm::RefVector<vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<vector<reco::Track>,reco::Track> >,reco::JTATagInfo>,reco::Vertex> > > (
  //                               edm::InputTag("hltCombinedSecondaryVertexBJetTagsPF"));
  pfTagsToken_              = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("onlineDiscrLabelPF"));
  caloTagsToken_            = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("onlineDiscrLabelCalo"));
  offlineDiscrTokenb_       = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("offlineDiscrLabelb"));
  offlineDiscrTokenbb_      = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("offlineDiscrLabelbb"));
  hltFastPVToken_           = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("hltFastPVLabel"));
  hltPFPVToken_             = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("hltPFPVLabel"));
  hltCaloPVToken_           = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("hltCaloPVLabel"));
  offlinePVToken_           = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("offlinePVLabel"));

  std::vector<edm::ParameterSet> paths =  iConfig.getParameter<std::vector<edm::ParameterSet> >("pathPairs");
  for(auto & path : paths) {
    custompathnamepairs_.push_back(make_pair(
					     path.getParameter<std::string>("pathname"),
					     path.getParameter<std::string>("pathtype")
					     ));}
}

BTVHLTOfflineSource::~BTVHLTOfflineSource() = default;


void BTVHLTOfflineSource::dqmBeginRun(const edm::Run& run, const edm::EventSetup& c)
{
  bool changed = true;
  if (!hltConfig_.init(run, c, processname_, changed)) {
  LogDebug("BTVHLTOfflineSource") << "HLTConfigProvider failed to initialize.";
  }

  for (unsigned int i=0; i!=hltConfig_.size(); ++i) {
    pathname_      = hltConfig_.triggerName(i);
    filtername_    = "dummy";
    unsigned int usedPrescale = 1;
    unsigned int objectType = 0;
    std::string triggerType = "";

    for (auto & custompathnamepair : custompathnamepairs_) {
      if(pathname_.find(custompathnamepair.first) != std::string::npos) {
        triggerType = custompathnamepair.second;
        hltPathsAll_.push_back(
          PathInfo(usedPrescale, pathname_, "dummy", processname_, objectType, triggerType));
      }
    }
  }
}


void
BTVHLTOfflineSource::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  iEvent.getByToken(triggerResultsToken, triggerResults_);
  if(!triggerResults_.isValid()) {
    iEvent.getByToken(triggerResultsFUToken,triggerResults_);
    if(!triggerResults_.isValid()) {
      edm::LogInfo("BTVHLTOfflineSource") << "TriggerResults not found, "
	"skipping event";
      return;
    }
  }

  triggerNames_ = iEvent.triggerNames(*triggerResults_);

  iEvent.getByToken(triggerSummaryToken,triggerObj_);
  if(!triggerObj_.isValid()) {
    iEvent.getByToken(triggerSummaryFUToken,triggerObj_);
    if(!triggerObj_.isValid()) {
      edm::LogInfo("BTVHLTOfflineSource") << "TriggerEvent not found, "
	"skipping event";
      return;
    }
  }

  iEvent.getByToken(caloTagsToken_, caloTags);
  iEvent.getByToken(pfTagsToken_, pfTags);

  Handle<reco::VertexCollection> VertexHandler;

  Handle<reco::JetTagCollection> offlineJetTagHandlerb;
  iEvent.getByToken(offlineDiscrTokenb_, offlineJetTagHandlerb);

  Handle<reco::JetTagCollection> offlineJetTagHandlerbb;
  iEvent.getByToken(offlineDiscrTokenbb_, offlineJetTagHandlerbb);

  Handle<reco::VertexCollection> offlineVertexHandler;
  iEvent.getByToken(offlinePVToken_, offlineVertexHandler);

  if(verbose_ && iEvent.id().event()%10000==0)
    cout<<"Run = "<<iEvent.id().run()<<", LS = "<<iEvent.luminosityBlock()<<", Event = "<<iEvent.id().event()<<endl;

  if(!triggerResults_.isValid()) return;

  for(auto & v : hltPathsAll_) {
    unsigned index = triggerNames_.triggerIndex(v.getPath());
    if (!(index < triggerNames_.size())) {
      continue;
    }

    // PF and Calo btagging
    if (   (v.getTriggerType() == "PF"   &&   pfTags.isValid())
        || (v.getTriggerType() == "Calo" && caloTags.isValid() && !caloTags->empty()) )
    {
      const auto & iter = (v.getTriggerType() == "PF") ? pfTags->begin() : caloTags->begin();

      float Discr_online = iter->second;
      if (Discr_online<0) Discr_online = -0.05;

      v.Discr->Fill(Discr_online);
      v.Pt->Fill(iter->first->pt());
      v.Eta->Fill(iter->first->eta());

      if(offlineJetTagHandlerb.isValid()){
        for (auto const & iterOffb : *offlineJetTagHandlerb){
          float DR = reco::deltaR(iterOffb.first->eta(),iterOffb.first->phi(),iter->first->eta(),iter->first->phi());
          if (DR<0.3) {
            float Discr_offline = iterOffb.second;

            // offline probb and probbb must be added (if probbb isn't specified, it'll just use probb)
            if(offlineJetTagHandlerbb.isValid()){
              for (auto const & iterOffbb : *offlineJetTagHandlerbb){
                DR = reco::deltaR(iterOffbb.first->eta(),iterOffbb.first->phi(),iter->first->eta(),iter->first->phi());
                if (DR<0.3) {
                  Discr_offline += iterOffbb.second;
                  break;
                }
              }
            }

            if (Discr_offline<0) Discr_offline = -0.05;
            v.Discr_HLTvsRECO->Fill(Discr_online, Discr_offline);
            v.Discr_HLTMinusRECO->Fill(Discr_online - Discr_offline);

            v.Discr_turnon_loose .denominator->Fill(Discr_offline);
            v.Discr_turnon_medium.denominator->Fill(Discr_offline);
            v.Discr_turnon_tight .denominator->Fill(Discr_offline);

            if (Discr_online > turnon_threshold_loose_) v.Discr_turnon_loose .numerator->Fill(Discr_offline);
            if (Discr_online > turnon_threshold_medium_)v.Discr_turnon_medium.numerator->Fill(Discr_offline);
            if (Discr_online > turnon_threshold_tight_) v.Discr_turnon_tight .numerator->Fill(Discr_offline);

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
          v.PVz_HLTMinusRECO->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
        }
      }
    }

    // specific to Calo b-tagging
    if (caloTags.isValid() && v.getTriggerType() == "Calo" && !caloTags->empty()) {
      iEvent.getByToken(hltCaloPVToken_, VertexHandler);
      if (VertexHandler.isValid()) {
        v.fastPVz->Fill(VertexHandler->begin()->z());
	      if (offlineVertexHandler.isValid()) {
          v.fastPVz_HLTMinusRECO->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
        }
      }
    }


    // additional plots from tag info collections
    /////////////////////////////////////////////

    iEvent.getByToken(shallowTagInfosTokenPf_, shallowTagInfosPf);
    iEvent.getByToken(shallowTagInfosTokenCalo_, shallowTagInfosCalo);
    // iEvent.getByToken(pfTagInfosToken_, pfTagInfos);
    // iEvent.getByToken(caloTagInfosToken_, caloTagInfos);

    // first try to get info from shallowTagInfos ...
    if (   (v.getTriggerType() == "PF"   && shallowTagInfosPf.isValid())
        || (v.getTriggerType() == "Calo" && shallowTagInfosCalo.isValid()) )
    {
      const auto & shallowTagInfoCollection = (v.getTriggerType() == "PF") ? shallowTagInfosPf : shallowTagInfosCalo;
      for (const auto & shallowTagInfo : *shallowTagInfoCollection) {
        const auto & tagVars = shallowTagInfo.taggingVariables();

        // n secondary vertices and n selected tracks
        for (const auto & tagVar : tagVars.getList(reco::btau::jetNSecondaryVertices, false)) {
          v.n_vtx->Fill(tagVar);
        }
        for (const auto & tagVar : tagVars.getList(reco::btau::jetNSelectedTracks, false)) {
          v.n_sel_tracks->Fill(tagVar);}

        // impact parameter
        const auto & trackSip3dVal = tagVars.getList(reco::btau::trackSip3dVal, false);
        const auto & trackSip3dSig = tagVars.getList(reco::btau::trackSip3dSig, false);
        for (unsigned i_trk=0; i_trk < trackSip3dVal.size(); i_trk++) {
          float val = trackSip3dVal[i_trk];
          float sig = trackSip3dSig[i_trk];
          v.h_3d_ip_distance->Fill(val);
          v.h_3d_ip_error->Fill(val/sig);
          v.h_3d_ip_sig->Fill(sig);
        }

        // vertex mass and tracks per vertex
        for (const auto & tagVar : tagVars.getList(reco::btau::vertexMass, false)) {
          v.vtx_mass->Fill(tagVar);}
        for (const auto & tagVar : tagVars.getList(reco::btau::vertexNTracks, false)) {
          v.n_vtx_trks->Fill(tagVar);}

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

void
BTVHLTOfflineSource::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & run, edm::EventSetup const & c)
{
  iBooker.setCurrentFolder(dirname_);
  for(auto & v : hltPathsAll_){
    //
    std::string trgPathName = HLTConfigProvider::removeVersion(v.getPath());
    std::string subdirName  = dirname_ +"/"+ trgPathName + v.getTriggerType();
    std::string trigPath    = "("+trgPathName+")";
    iBooker.setCurrentFolder(subdirName);

    std::string labelname("HLT");
    std::string histoname(labelname+"");
    std::string title(labelname+"");

    histoname = labelname+"_Discr";
    title = labelname+"_Discr "+trigPath;
    v.Discr =  iBooker.book1D(histoname.c_str(),title.c_str(),110,-0.1,1);

    histoname = labelname+"_Pt";
    title = labelname+"_Pt "+trigPath;
    v.Pt =  iBooker.book1D(histoname.c_str(),title.c_str(),100,0,400);

    histoname = labelname+"_Eta";
    title = labelname+"_Eta "+trigPath;
    v.Eta =  iBooker.book1D(histoname.c_str(),title.c_str(),60,-3.0,3.0);

    histoname = "HLTvsRECO_Discr";
    title = "online discr vs offline discr "+trigPath;
    v.Discr_HLTvsRECO =  iBooker.book2D(histoname.c_str(),title.c_str(),110,-0.1,1,110,-0.1,1);

    histoname = "HLTMinusRECO_Discr";
    title = "online discr minus offline discr "+trigPath;
    v.Discr_HLTMinusRECO =  iBooker.book1D(histoname.c_str(),title.c_str(),100,-1,1);

    histoname = "Turnon_loose_Discr";
    title = "turn-on with loose threshold "+trigPath;
    v.bookME(iBooker, v.Discr_turnon_loose, histoname, title, 22, -0.1, 1.);

    histoname = "Turnon_medium_Discr";
    title = "turn-on with medium threshold "+trigPath;
    v.bookME(iBooker, v.Discr_turnon_medium, histoname, title, 22, -0.1, 1.);

    histoname = "Turnon_tight_Discr";
    title = "turn-on with tight threshold "+trigPath;
    v.bookME(iBooker, v.Discr_turnon_tight, histoname, title, 22, -0.1, 1.);

    histoname = labelname+"_PVz";
    title = "online z(PV) "+trigPath;
    v.PVz =  iBooker.book1D(histoname.c_str(),title.c_str(),80,-20,20);

    histoname = labelname+"_fastPVz";
    title = "online z(fastPV) "+trigPath;
    v.fastPVz =  iBooker.book1D(histoname.c_str(),title.c_str(),80,-20,20);

    histoname = "HLTMinusRECO_PVz";
    title = "online z(PV) - offline z(PV) "+trigPath;
    v.PVz_HLTMinusRECO =  iBooker.book1D(histoname.c_str(),title.c_str(),200,-0.5,0.5);

    histoname = "HLTMinusRECO_fastPVz";
    title = "online z(fastPV) - offline z(PV) "+trigPath;
    v.fastPVz_HLTMinusRECO =  iBooker.book1D(histoname.c_str(),title.c_str(),100,-2,2);

    histoname = "n_vtx";
    title = "N vertex candidates "+trigPath;
    v.n_vtx =  iBooker.book1D(histoname.c_str(),title.c_str(), 10, -0.5, 9.5);

    histoname = "vtx_mass";
    title = "secondary vertex mass (GeV)"+trigPath;
    v.vtx_mass = iBooker.book1D(histoname.c_str(), title.c_str(), 20, 0, 10);

    histoname = "n_vtx_trks";
    title = "N tracks associated to secondary vertex"+trigPath;
    v.n_vtx_trks = iBooker.book1D(histoname.c_str(), title.c_str(), 20, -0.5, 19.5);

    histoname = "n_sel_tracks";
    title = "N selected tracks"+trigPath;
    v.n_sel_tracks = iBooker.book1D(histoname.c_str(), title.c_str(), 25, -0.5, 24.5);

    histoname = "3d_ip_distance";
    title = "3D IP distance of tracks (cm)"+trigPath;
    v.h_3d_ip_distance = iBooker.book1D(histoname.c_str(), title.c_str(), 40, -0.1, 0.1);

    histoname = "3d_ip_error";
    title = "3D IP error of tracks (cm)"+trigPath;
    v.h_3d_ip_error = iBooker.book1D(histoname.c_str(), title.c_str(), 40, 0., 0.1);

    histoname = "3d_ip_sig";
    title = "3D IP significance of tracks (cm)"+trigPath;
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
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BTVHLTOfflineSource);
