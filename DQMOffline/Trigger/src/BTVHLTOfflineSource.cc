#include "DQMOffline/Trigger/interface/BTVHLTOfflineSource.h"

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

  dirname_                = iConfig.getUntrackedParameter("dirname",std::string("HLT/BTV/"));
  processname_            = iConfig.getParameter<std::string>("processname");
  verbose_                = iConfig.getUntrackedParameter< bool >("verbose", false);
  triggerSummaryLabel_    = iConfig.getParameter<edm::InputTag>("triggerSummaryLabel");
  triggerResultsLabel_    = iConfig.getParameter<edm::InputTag>("triggerResultsLabel");
  triggerSummaryToken     = consumes <trigger::TriggerEvent> (triggerSummaryLabel_);
  triggerResultsToken     = consumes <edm::TriggerResults>   (triggerResultsLabel_);
  triggerSummaryFUToken   = consumes <trigger::TriggerEvent> (edm::InputTag(triggerSummaryLabel_.label(),triggerSummaryLabel_.instance(),std::string("FU")));
  triggerResultsFUToken   = consumes <edm::TriggerResults>   (edm::InputTag(triggerResultsLabel_.label(),triggerResultsLabel_.instance(),std::string("FU")));
  csvCaloTagInfosToken_   = consumes<vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<edm::RefVector<vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<vector<reco::Track>,reco::Track> >,reco::JTATagInfo>,reco::Vertex> > > (
                                edm::InputTag("hltInclusiveSecondaryVertexFinderTagInfos"));
  csvPfTagInfosToken_     = consumes<vector<reco::TemplatedSecondaryVertexTagInfo<reco::IPTagInfo<edm::RefVector<vector<reco::Track>,reco::Track,edm::refhelper::FindUsingAdvance<vector<reco::Track>,reco::Track> >,reco::JTATagInfo>,reco::Vertex> > > (
                                edm::InputTag("hltSecondaryVertexTagInfosPF"));
  csvCaloTagsToken_       = consumes<reco::JetTagCollection> (edm::InputTag("hltCombinedSecondaryVertexBJetTagsCalo"));
  csvPfTagsToken_         = consumes<reco::JetTagCollection> (edm::InputTag("hltCombinedSecondaryVertexBJetTagsPF"));
  offlineCSVTokenPF_      = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("offlineCSVLabelPF"));
  offlineCSVTokenCalo_    = consumes<reco::JetTagCollection> (iConfig.getParameter<edm::InputTag>("offlineCSVLabelCalo"));
  hltFastPVToken_         = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("hltFastPVLabel"));
  hltPFPVToken_           = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("hltPFPVLabel"));
  hltCaloPVToken_         = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("hltCaloPVLabel"));
  offlinePVToken_         = consumes<std::vector<reco::Vertex> > (iConfig.getParameter<edm::InputTag>("offlinePVLabel"));

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
    bool changed(true);
    if (!hltConfig_.init(run, c, processname_, changed)) {
    LogDebug("BTVHLTOfflineSource") << "HLTConfigProvider failed to initialize.";
    }

  const unsigned int numberOfPaths(hltConfig_.size());
  for(unsigned int i=0; i!=numberOfPaths; ++i){
    pathname_      = hltConfig_.triggerName(i);
    filtername_    = "dummy";
    unsigned int usedPrescale = 1;
    unsigned int objectType = 0;
    std::string triggerType = "";
    bool trigSelected = false;

    for (auto & custompathnamepair : custompathnamepairs_){
       if(pathname_.find(custompathnamepair.first)!=std::string::npos) { trigSelected = true; triggerType = custompathnamepair.second;}
      }

    if (!trigSelected) continue;

    hltPathsAll_.push_back(PathInfo(usedPrescale, pathname_, "dummy", processname_, objectType, triggerType));
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

  iEvent.getByToken(csvCaloTagsToken_, csvCaloTags);
  iEvent.getByToken(csvPfTagsToken_, csvPfTags);

  Handle<reco::VertexCollection> VertexHandler;

  Handle<reco::JetTagCollection> offlineJetTagHandlerPF;
  iEvent.getByToken(offlineCSVTokenPF_, offlineJetTagHandlerPF);

  Handle<reco::JetTagCollection> offlineJetTagHandlerCalo;
  iEvent.getByToken(offlineCSVTokenCalo_, offlineJetTagHandlerCalo);

  Handle<reco::VertexCollection> offlineVertexHandler;
  iEvent.getByToken(offlinePVToken_, offlineVertexHandler);

  if(verbose_ && iEvent.id().event()%10000==0)
    cout<<"Run = "<<iEvent.id().run()<<", LS = "<<iEvent.luminosityBlock()<<", Event = "<<iEvent.id().event()<<endl;

  if(!triggerResults_.isValid()) return;

  for(auto & v : hltPathsAll_){
    unsigned index = triggerNames_.triggerIndex(v.getPath());
    if (index < triggerNames_.size() ){
     float DR  = 9999.;
//                                                                                _|_|_|    _|_|_|_|
//                                                                                _|    _|  _|
//                                                                                _|_|_|    _|_|_|
//                                                                                _|        _|
//                                                                                _|        _|
     if (csvPfTags.isValid() && v.getTriggerType() == "PF")
     {
      auto iter = csvPfTags->begin();

      float CSV_online = iter->second;
      if (CSV_online<0) CSV_online = -0.05;

      v.getMEhisto_CSV()->Fill(CSV_online);
      v.getMEhisto_Pt()->Fill(iter->first->pt());
      v.getMEhisto_Eta()->Fill(iter->first->eta());

      DR  = 9999.;
      if(offlineJetTagHandlerPF.isValid()){
          for (auto const & iterO : *offlineJetTagHandlerPF){
            float CSV_offline = iterO.second;
            if (CSV_offline<0) CSV_offline = -0.05;
            DR = reco::deltaR(iterO.first->eta(),iterO.first->phi(),iter->first->eta(),iter->first->phi());
            if (DR<0.3) {
               v.getMEhisto_CSV_RECOvsHLT()->Fill(CSV_offline,CSV_online); continue;
               }
          }
      }

      iEvent.getByToken(hltPFPVToken_, VertexHandler);
      if (VertexHandler.isValid())
      {
        v.getMEhisto_PVz()->Fill(VertexHandler->begin()->z());
        if (offlineVertexHandler.isValid()) v.getMEhisto_PVz_HLTMinusRECO()->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
        }
      }

      iEvent.getByToken(csvPfTagInfosToken_, csvPfTagInfos);
      if (csvPfTagInfos.isValid() && v.getTriggerType() == "PF") {

        // loop over secondary vertex tag infos
        for (const auto & csvPfTagInfo : *csvPfTagInfos) {
          v.getMEhisto_n_vtx()->Fill(csvPfTagInfo.nVertexCandidates());
          v.getMEhisto_n_sel_tracks()->Fill(csvPfTagInfo.nSelectedTracks());

          // loop over selected tracks in each tag info
          for (unsigned i_trk=0; i_trk < csvPfTagInfo.nSelectedTracks(); i_trk++) {
            const auto & ip3d = csvPfTagInfo.trackIPData(i_trk).ip3d;
            v.getMEhisto_h_3d_ip_distance()->Fill(ip3d.value());
            v.getMEhisto_h_3d_ip_error()->Fill(ip3d.error());
            v.getMEhisto_h_3d_ip_sig()->Fill(ip3d.significance());
          }

          // loop over vertex candidates in each tag info
          for (unsigned i_sv=0; i_sv < csvPfTagInfo.nVertexCandidates(); i_sv++) {
            const auto & sv = csvPfTagInfo.secondaryVertex(i_sv);
            v.getMEhisto_vtx_mass()->Fill(sv.p4().mass());
            v.getMEhisto_n_vtx_trks()->Fill(sv.nTracks());

            // loop over tracks for number of pixel and total hits
            const auto & trkIPTagInfo = csvPfTagInfo.trackIPTagInfoRef().get();
            for (const auto & trk : trkIPTagInfo->selectedTracks()) {
              v.getMEhisto_n_pixel_hits()->Fill(trk.get()->hitPattern().numberOfValidPixelHits());
              v.getMEhisto_n_total_hits()->Fill(trk.get()->hitPattern().numberOfValidHits());
            }
          }
        }
      }

//                                                                    _|_|_|            _|
//                                                                  _|          _|_|_|  _|    _|_|
//                                                                  _|        _|    _|  _|  _|    _|
//                                                                  _|        _|    _|  _|  _|    _|
//                                                                    _|_|_|    _|_|_|  _|    _|_|

     if (csvCaloTags.isValid() && v.getTriggerType() == "Calo" && !csvCaloTags->empty())
     {

      auto iter = csvCaloTags->begin();

      float CSV_online = iter->second;
      if (CSV_online<0) CSV_online = -0.05;

      v.getMEhisto_CSV()->Fill(CSV_online);
      v.getMEhisto_Pt()->Fill(iter->first->pt());
      v.getMEhisto_Eta()->Fill(iter->first->eta());

      DR  = 9999.;
      if(offlineJetTagHandlerCalo.isValid()){
          for (auto const & iterO : *offlineJetTagHandlerCalo)
          {
            float CSV_offline = iterO.second;
            if (CSV_offline<0) CSV_offline = -0.05;
            DR = reco::deltaR(iterO.first->eta(),iterO.first->phi(),iter->first->eta(),iter->first->phi());
            if (DR<0.3)
            {
                v.getMEhisto_CSV_RECOvsHLT()->Fill(CSV_offline,CSV_online); continue;
            }
          }
      }

      iEvent.getByToken(hltFastPVToken_, VertexHandler);
      if (VertexHandler.isValid())
      {
        v.getMEhisto_PVz()->Fill(VertexHandler->begin()->z());
	if (offlineVertexHandler.isValid()) v.getMEhisto_fastPVz_HLTMinusRECO()->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
       }

      iEvent.getByToken(hltCaloPVToken_, VertexHandler);
      if (VertexHandler.isValid())
      {
        v.getMEhisto_fastPVz()->Fill(VertexHandler->begin()->z());
	if (offlineVertexHandler.isValid()) v.getMEhisto_PVz_HLTMinusRECO()->Fill(VertexHandler->begin()->z()-offlineVertexHandler->begin()->z());
       }

      }

      iEvent.getByToken(csvCaloTagInfosToken_, csvCaloTagInfos);
      if (csvCaloTagInfos.isValid() && v.getTriggerType() == "Calo") {

        // loop over secondary vertex tag infos
        for (const auto & csvCaloTagInfo : *csvCaloTagInfos) {
          v.getMEhisto_n_vtx()->Fill(csvCaloTagInfo.nVertexCandidates());
          v.getMEhisto_n_sel_tracks()->Fill(csvCaloTagInfo.nSelectedTracks());

          // loop over selected tracks in each tag info
          for (unsigned i_trk=0; i_trk < csvCaloTagInfo.nSelectedTracks(); i_trk++) {
            const auto & ip3d = csvCaloTagInfo.trackIPData(i_trk).ip3d;
            v.getMEhisto_h_3d_ip_distance()->Fill(ip3d.value());
            v.getMEhisto_h_3d_ip_error()->Fill(ip3d.error());
            v.getMEhisto_h_3d_ip_sig()->Fill(ip3d.significance());
          }

          // loop over vertex candidates in each tag info
          for (unsigned i_sv=0; i_sv < csvCaloTagInfo.nVertexCandidates(); i_sv++) {
            const auto & sv = csvCaloTagInfo.secondaryVertex(i_sv);
            v.getMEhisto_vtx_mass()->Fill(sv.p4().mass());
            v.getMEhisto_n_vtx_trks()->Fill(sv.nTracks());

            // loop over tracks for number of pixel and total hits
            const auto & trkIPTagInfo = csvCaloTagInfo.trackIPTagInfoRef().get();
            for (const auto & trk : trkIPTagInfo->selectedTracks()) {
              v.getMEhisto_n_pixel_hits()->Fill(trk.get()->hitPattern().numberOfValidPixelHits());
              v.getMEhisto_n_total_hits()->Fill(trk.get()->hitPattern().numberOfValidHits());
            }
          }
        }
      }


    }
   }
}

void
BTVHLTOfflineSource::bookHistograms(DQMStore::IBooker & iBooker, edm::Run const & run, edm::EventSetup const & c)
{
   iBooker.setCurrentFolder(dirname_);
   for(auto & v : hltPathsAll_){
     //
     std::string trgPathName = HLTConfigProvider::removeVersion(v.getPath());
     std::string subdirName  = dirname_ +"/"+ trgPathName;
     std::string trigPath    = "("+trgPathName+")";
     iBooker.setCurrentFolder(subdirName);

     std::string labelname("HLT");
     std::string histoname(labelname+"");
     std::string title(labelname+"");

     histoname = labelname+"_CSV";
     title = labelname+"_CSV "+trigPath;
     MonitorElement * CSV =  iBooker.book1D(histoname.c_str(),title.c_str(),110,-0.1,1);

     histoname = labelname+"_Pt";
     title = labelname+"_Pt "+trigPath;
     MonitorElement * Pt =  iBooker.book1D(histoname.c_str(),title.c_str(),100,0,400);

     histoname = labelname+"_Eta";
     title = labelname+"_Eta "+trigPath;
     MonitorElement * Eta =  iBooker.book1D(histoname.c_str(),title.c_str(),60,-3.0,3.0);

     histoname = "RECOvsHLT_CSV";
     title = "offline CSV vs online CSV "+trigPath;
     MonitorElement * CSV_RECOvsHLT =  iBooker.book2D(histoname.c_str(),title.c_str(),110,-0.1,1,110,-0.1,1);

     histoname = labelname+"_PVz";
     title = "online z(PV) "+trigPath;
     MonitorElement * PVz =  iBooker.book1D(histoname.c_str(),title.c_str(),80,-20,20);

     histoname = labelname+"_fastPVz";
     title = "online z(fastPV) "+trigPath;
     MonitorElement * fastPVz =  iBooker.book1D(histoname.c_str(),title.c_str(),80,-20,20);

     histoname = "HLTMinusRECO_PVz";
     title = "online z(PV) - offline z(PV) "+trigPath;
     MonitorElement * PVz_HLTMinusRECO =  iBooker.book1D(histoname.c_str(),title.c_str(),200,-0.5,0.5);

     histoname = "HLTMinusRECO_fastPVz";
     title = "online z(fastPV) - offline z(PV) "+trigPath;
     MonitorElement * fastPVz_HLTMinusRECO =  iBooker.book1D(histoname.c_str(),title.c_str(),100,-2,2);

     histoname = "n_vtx";
     title = "N vertex candidates "+trigPath;
     MonitorElement * n_vtx =  iBooker.book1D(histoname.c_str(),title.c_str(), 10, -0.5, 9.5);

     histoname = "vtx_mass";
     title = "secondary vertex mass (GeV)"+trigPath;
     MonitorElement * vtx_mass = iBooker.book1D(histoname.c_str(), title.c_str(), 20, 0, 10);

     histoname = "n_vtx_trks";
     title = "N tracks associated to secondary vertex"+trigPath;
     MonitorElement * n_vtx_trks = iBooker.book1D(histoname.c_str(), title.c_str(), 20, -0.5, 19.5);

     histoname = "n_sel_tracks";
     title = "N selected tracks"+trigPath;
     MonitorElement * n_sel_tracks = iBooker.book1D(histoname.c_str(), title.c_str(), 25, -0.5, 24.5);

     histoname = "3d_ip_distance";
     title = "3D IP distance of tracks (cm)"+trigPath;
     MonitorElement * h_3d_ip_distance = iBooker.book1D(histoname.c_str(), title.c_str(), 40, -0.1, 0.1);

     histoname = "3d_ip_error";
     title = "3D IP error of tracks (cm)"+trigPath;
     MonitorElement * h_3d_ip_error = iBooker.book1D(histoname.c_str(), title.c_str(), 40, 0., 0.1);

     histoname = "3d_ip_sig";
     title = "3D IP significance of tracks (cm)"+trigPath;
     MonitorElement * h_3d_ip_sig = iBooker.book1D(histoname.c_str(), title.c_str(), 40, -40, 40);

     histoname = "n_pixel_hits";
     title = "N pixel hits"+trigPath;
     MonitorElement * n_pixel_hits = iBooker.book1D(histoname.c_str(), title.c_str(), 16, -0.5, 15.5);

     histoname = "n_total_hits";
     title = "N hits"+trigPath;
     MonitorElement * n_total_hits = iBooker.book1D(histoname.c_str(), title.c_str(), 40, -0.5, 39.5);


     v.setHistos(
        CSV,Pt,Eta,CSV_RECOvsHLT,PVz,fastPVz,PVz_HLTMinusRECO,fastPVz_HLTMinusRECO,
        n_vtx, vtx_mass, n_vtx_trks, n_sel_tracks, h_3d_ip_distance, h_3d_ip_error, h_3d_ip_sig, n_pixel_hits, n_total_hits
     );
   }
}
