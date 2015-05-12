// -*- C++ -*-
//
// Package:    DQM/HLTWorkspace
// Class:      HLTWorkspace
//
/**\class HLTWorkspace HLTWorkspace.cc DQM/HLTEvF/plugins/HLTWorkspace.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Charles Nicholas Mueller
//         Created:  Sun, 22 Mar 2015 22:29:00 GMT
//
//


// system include files
#include <memory>
#include <sys/time.h>
#include <cstdlib>

// user include files
#include "DataFormats/Common/interface/TriggerResults.h"
#include "DataFormats/HLTReco/interface/TriggerObject.h"
#include "DataFormats/HLTReco/interface/TriggerEvent.h"
#include "DataFormats/HLTReco/interface/TriggerTypeDefs.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"

//for collections
#include "HLTrigger/JetMET/interface/AlphaT.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/SiStripCluster/interface/SiStripCluster.h"
#include "CommonTools/RecoAlgos/interface/TrackSelector.h"


#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "TMath.h"
#include "TStyle.h"
#include "TLorentzVector.h"

//
// class declaration
//

//using namespace edm;
using namespace trigger;
using std::vector;
using std::string;
using std::map;

class HLTWorkspace : public DQMEDAnalyzer {
   public:
      explicit HLTWorkspace(const edm::ParameterSet&);
      ~HLTWorkspace();

  //      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void bookHistograms(DQMStore::IBooker &i, edm::Run const&, edm::EventSetup const&) override;
      void dqmBeginRun(edm::Run const&, edm::EventSetup const&) override;

      void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;

  void fillPlots(int, string, edm::Handle<trigger::TriggerEvent>);
  double get_wall_time(void);
      // ----------member data ---------------------------

  bool debugPrint;

  HLTConfigProvider hltConfig_;

  string topDirectoryName;
  string mainShifterFolder;
  string backupFolder;

  map<string, unsigned int> lookupIndex;
  map<string, string> lookupFilter;
  vector<string> quickCollectionPaths;

  //set Token(-s)
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> aodTriggerToken_;
  edm::EDGetTokenT<LumiScalersCollection> lumiScalersToken_;

  edm::EDGetTokenT<edmNew::DetSetVector<SiPixelCluster>> siPixelClusterToken_;
  edm::EDGetTokenT<edmNew::DetSetVector<SiStripCluster>> siStripClusterToken_;
  edm::EDGetTokenT<TrackingRecHitCollection> trackingRecHitsToken_;  
  edm::EDGetTokenT<reco::TrackExtraCollection> trackExtraToken_;  
  edm::EDGetTokenT<reco::TrackCollection> trackToken_;  
  //  edm::EDGetTokenT<reco::BeamSpot> trackToken_;  

  //  edm::EDGetTokenT<reco::JetTagCollection> csvTagToken_;
  
  //declare all MEs
  MonitorElement * photonPt_;
  MonitorElement * photonEta_;
  MonitorElement * photonPhi_;
  MonitorElement * muonPt_;
  MonitorElement * muonEta_;
  MonitorElement * muonPhi_;
  MonitorElement * electronPt_;
  MonitorElement * electronEta_;
  MonitorElement * electronPhi_;
  MonitorElement * jetPt_;
  MonitorElement * tauPt_;
  MonitorElement * diMuonLowMass_;
  MonitorElement * alphaT_;
  MonitorElement * caloMetPt_;
  MonitorElement * caloMetPhi_;
  MonitorElement * pfMetPt_;
  MonitorElement * pfMetPhi_;
  MonitorElement * caloHtPt_;
  MonitorElement * pfHtPt_;
  MonitorElement * bJetPhi_;
  MonitorElement * bJetEta_;

  MonitorElement * wallTimePerEvent_;

};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
HLTWorkspace::HLTWorkspace(const edm::ParameterSet& iConfig)

{
   //now do what ever initialization is needed
  debugPrint = false;

  topDirectoryName = "HLT/Workspaces";
  mainShifterFolder = topDirectoryName+"/MainShifter";
  backupFolder = topDirectoryName+"/Backup";

  //set Token(s) 
  //will need to change 'TEST' to 'HLT' or something else before implementation
  triggerResultsToken_ = consumes<edm::TriggerResults>(edm::InputTag("TriggerResults","", "TEST"));
  aodTriggerToken_ = consumes<trigger::TriggerEvent>(edm::InputTag("hltTriggerSummaryAOD", "", "TEST"));
  lumiScalersToken_ = consumes<LumiScalersCollection>(edm::InputTag("hltScalersRawToDigi","",""));
  siPixelClusterToken_ = consumes<edmNew::DetSetVector<SiPixelCluster>>(edm::InputTag("hltSiPixelClusters","","TEST"));
  siStripClusterToken_ = consumes<edmNew::DetSetVector<SiStripCluster>>(edm::InputTag("hltSiStripRawToClustersFacility","","TEST"));
  trackingRecHitsToken_ = consumes<TrackingRecHitCollection>(edm::InputTag("hltIter2Merged","","TEST"));
  trackExtraToken_ = consumes<reco::TrackExtraCollection>(edm::InputTag("hltIter2Merged","","TEST"));
  trackToken_ = consumes<reco::TrackCollection>(edm::InputTag("hltIter2Merged","","TEST"));
  
  

  // use this csvTagToken_ = consumes<reco::JetTagCollection>(InputTag("hltCombinedSecondaryVertexBJetTagsPF","","TEST")); 
  // prob not this csvTagToken_ = consumes<edm::AssociationVector<edm::RefToBaseProd<reco::Jet>,vector<float>,edm::RefToBase<reco::Jet>,unsigned int,edm::helper::AssociationIdenticalKeyReference>>(InputTag("hltCombinedSecondaryVertexBJetTagsPF","","TEST")); 

}


HLTWorkspace::~HLTWorkspace()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

// ------------ method called for each event  ------------
void
HLTWorkspace::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
  double start = get_wall_time();
  
  using namespace edm;
  
   if (debugPrint) std::cout << "Inside analyze(). " << std::endl;
   int eventNumber = iEvent.id().event();

   // access trigger results
   edm::Handle<edm::TriggerResults> triggerResults;
   iEvent.getByToken(triggerResultsToken_, triggerResults);
   if (!triggerResults.isValid()) return;

   edm::Handle<trigger::TriggerEvent> aodTriggerEvent;
   iEvent.getByToken(aodTriggerToken_, aodTriggerEvent);
   if (!aodTriggerEvent.isValid()) return;

   edm::Handle<edmNew::DetSetVector<SiPixelCluster>> siPixelCluster;
   iEvent.getByToken(siPixelClusterToken_, siPixelCluster);
   if (!siPixelCluster.isValid()) return;

   edm::Handle<edmNew::DetSetVector<SiStripCluster>> siStripCluster;
   iEvent.getByToken(siStripClusterToken_, siStripCluster);
   if (!siStripCluster.isValid()) return;

   edm::Handle<TrackingRecHitCollection> trackingRecHits;
   iEvent.getByToken(trackingRecHitsToken_, trackingRecHits);
   if (!trackingRecHits.isValid()) return;

   edm::Handle<reco::TrackExtraCollection> trackExtras;
   iEvent.getByToken(trackExtraToken_, trackExtras);
   if (!trackExtras.isValid()) return;

   edm::Handle<reco::TrackCollection> tracks;
   iEvent.getByToken(trackToken_, tracks);
   if (!tracks.isValid()) return;

   

   // edm::Handle<reco::JetTagCollection> jetTags;
   // iEvent.getByToken(csvTagToken_, jetTags);
   // if (!jetTags.isValid()) return;

   for (string & pathName : quickCollectionPaths) //loop over paths
     {
       if (triggerResults->accept(lookupIndex[pathName]) && hltConfig_.saveTags(lookupFilter[pathName]))
	 {
	   fillPlots(eventNumber, pathName, aodTriggerEvent);
	 }
     }

   //   sleep(1); //sleep for 1s, used to calibrate timing 
   double end = get_wall_time();
   double wallTime = end - start;
   wallTimePerEvent_->Fill(wallTime);
}

// ------------ method called when starting to processes a run  ------------
void
HLTWorkspace::dqmBeginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  if (debugPrint) std::cout << "Calling beginRun. " << std::endl;
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, "TEST", changed))
    {
      if (debugPrint) std::cout << "Extracting HLTconfig. " << std::endl;
    }

  //link all paths and filters needed
  quickCollectionPaths.push_back("HLT_Photon30_R9Id90_HE10_IsoM");
  lookupFilter["HLT_Photon30_R9Id90_HE10_IsoM"] = "hltEG30R9Id90HE10IsoMTrackIsoFilter";
  
  quickCollectionPaths.push_back("HLT_IsoMu27");
  lookupFilter["HLT_IsoMu27"] = "hltL3crIsoL1sMu25L1f0L2f10QL3f27QL3trkIsoFiltered0p09";

  quickCollectionPaths.push_back("HLT_Ele27_eta2p1_WP75_Gsf");
  lookupFilter["HLT_Ele27_eta2p1_WP75_Gsf"] = "hltEle27WP75GsfTrackIsoFilter";

  quickCollectionPaths.push_back("HLT_DoubleMu4_3_Jpsi_Displaced");
  lookupFilter["HLT_DoubleMu4_3_Jpsi_Displaced"] = "hltDisplacedmumuFilterDoubleMu43Jpsi";

  quickCollectionPaths.push_back("HLT_PFHT200_DiPFJet90_PFAlphaT0p57");
  lookupFilter["HLT_PFHT200_DiPFJet90_PFAlphaT0p57"] = "hltPFHT200PFAlphaT0p57";

  quickCollectionPaths.push_back("HLT_PFJet200");
  lookupFilter["HLT_PFJet200"] = "hltSinglePFJet200";

  quickCollectionPaths.push_back("HLT_MET75_IsoTrk50");
  lookupFilter["HLT_MET75_IsoTrk50"] = "hltMETClean75";

  quickCollectionPaths.push_back("HLT_PFHT750_4Jet");
  lookupFilter["HLT_PFHT750_4Jet"] = "hltPF4JetHT750";

  quickCollectionPaths.push_back("HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg");
  lookupFilter["HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg"] = "hltDoublePFTau40TrackPt1MediumIsolationDz02Reg";

  quickCollectionPaths.push_back("HLT_PFMET120_PFMHT120_IDTight");
  lookupFilter["HLT_PFMET120_PFMHT120_IDTight"] = "hltPFMET120";

  quickCollectionPaths.push_back("HLT_HT650_DisplacedDijet80_Inclusive");
  lookupFilter["HLT_HT650_DisplacedDijet80_Inclusive"] = "hltHT650";

  quickCollectionPaths.push_back("HLT_PFMET120_NoiseCleaned_BTagCSV07");
  lookupFilter["HLT_PFMET120_NoiseCleaned_BTagCSV07"] = "hltPFMET120Filter";

  quickCollectionPaths.push_back(" HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq500");
  lookupFilter[" HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq500"] = "hltCSVPF0p7";

  // quickCollectionPaths.push_back("");
  // lookupFilter[""] = "";

  string pathName_noVersion;
  vector<string> datasetPaths;

  vector<string> datasetNames = hltConfig_.streamContent("A");
  for (unsigned int i=0;i<datasetNames.size();i++) {
    datasetPaths = hltConfig_.datasetContent(datasetNames[i]);
    for (const auto & pathName : datasetPaths){
      pathName_noVersion = hltConfig_.removeVersion(pathName);
      //only add unique pathNames (keys) to the lookup table
      if (lookupIndex.find(pathName_noVersion) == lookupIndex.end()){
	lookupIndex[pathName_noVersion] = hltConfig_.triggerIndex(pathName);
      }
    }
  }

}

// ------------ method called when ending the processing of a run  ------------

void
HLTWorkspace::endRun(edm::Run const&, edm::EventSetup const&)
{
  if (debugPrint) std::cout << "Calling endRun. " << std::endl;
}

//void HLTWorkspace::bookPlots()
void HLTWorkspace::bookHistograms(DQMStore::IBooker & ibooker, edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  
  ////////////////////////////////
  ///
  /// Main shifter workspace plots
  ///
  ////////////////////////////////
  ibooker.setCurrentFolder(mainShifterFolder);

  //wall time
  TH1F * hist_wallTime = new TH1F("wallTime","wall time per event",1000,0,3);
  hist_wallTime->SetMinimum(0);
  wallTimePerEvent_ = ibooker.book1D("wallTime",hist_wallTime);

  //photon pt
  TH1F * hist_photonPt = new TH1F("Photon_pT","Photon pT",100,0,200);
  hist_photonPt->SetMinimum(0);
  photonPt_ = ibooker.book1D("Photon_pT",hist_photonPt);

  //muon pt
  TH1F * hist_muonPt = new TH1F("Muon_pT","Muon pT",75,0,150);
  hist_muonPt->SetMinimum(0);
  muonPt_ = ibooker.book1D("Muon_pT",hist_muonPt);

  //electron pt
  TH1F * hist_electronPt = new TH1F("Electron_pT","Electron pT",75,0,150);
  hist_electronPt->SetMinimum(0);
  electronPt_ = ibooker.book1D("Electron_pT",hist_electronPt);

  //jet pt
  TH1F * hist_jetPt = new TH1F("Jet_pT","Jet pT",75,150,550);
  hist_jetPt->SetMinimum(0);
  jetPt_ = ibooker.book1D("Jet_pT",hist_jetPt);

  //tau pt
  TH1F * hist_tauPt = new TH1F("Tau_pT","Tau pT",75,30,350);
  hist_tauPt->SetMinimum(0);
  tauPt_ = ibooker.book1D("Tau_pT",hist_tauPt);

  //dimuon low mass
  TH1F * hist_dimuonLowMass = new TH1F("Dimuon_LowMass","Dimuon Low Mass",100,2.5,3.5);
  hist_dimuonLowMass->SetMinimum(0);
  diMuonLowMass_ = ibooker.book1D("Dimuon_LowMass",hist_dimuonLowMass);

  //alphaT
  TH1F * hist_alphaT = new TH1F("alphaT","alphaT",30,0,5);
  hist_alphaT->SetMinimum(0);
  alphaT_ = ibooker.book1D("AlphaT",hist_alphaT);

  //caloMET pt
  TH1F * hist_caloMetPt = new TH1F("CaloMET_pT","CaloMET pT",60,50,550);
  hist_caloMetPt->SetMinimum(0);
  caloMetPt_ = ibooker.book1D("CaloMET_pT",hist_caloMetPt);

  //caloHT pt
  TH1F * hist_caloHtPt = new TH1F("CaloHT_pT","CaloHT pT",200,0,2000);
  hist_caloHtPt->SetMinimum(0);
  caloHtPt_ = ibooker.book1D("CaloHT_pT",hist_caloHtPt);

  //PFHT pt
  TH1F * hist_pfHtPt = new TH1F("PFHT_pT","PFHT pT",200,0,2000);
  hist_pfHtPt->SetMinimum(0);
  pfHtPt_ = ibooker.book1D("PFHT_pT",hist_pfHtPt);

  //PFMET pt
  TH1F * hist_PFMetPt = new TH1F("PFMET_pT","PFMET pT",60,100,500);
  hist_PFMetPt->SetMinimum(0);
  pfMetPt_ = ibooker.book1D("PFMET_pT",hist_PFMetPt);


  ////////////////////////////////
  ///
  /// Backup workspace plots
  ///
  ////////////////////////////////
  ibooker.setCurrentFolder(backupFolder);

  //photon eta
  TH1F * hist_photonEta = new TH1F("Photon_eta","Photon eta",50,0,3);
  hist_photonEta->SetMinimum(0);
  photonEta_ = ibooker.book1D("Photon_eta",hist_photonEta);
  //photon phi
  TH1F * hist_photonPhi = new TH1F("Photon_phi","Photon phi",50,-3.4,3.4);
  hist_photonPhi->SetMinimum(0);
  photonPhi_ = ibooker.book1D("Photon_phi",hist_photonPhi);

  //muon eta
  TH1F * hist_muonEta = new TH1F("Muon_eta","Muon eta",50,0,3);
  hist_muonEta->SetMinimum(0);
  muonEta_ = ibooker.book1D("Muon_eta",hist_muonEta);
  //muon phi
  TH1F * hist_muonPhi = new TH1F("Muon_phi","Muon phi",50,-3.4,3.4);
  hist_muonPhi->SetMinimum(0);
  muonPhi_ = ibooker.book1D("Muon_phi",hist_muonPhi);

  //electron eta
  TH1F * hist_electronEta = new TH1F("Electron_eta","Electron eta",50,0,3);
  hist_electronEta->SetMinimum(0);
  electronEta_ = ibooker.book1D("Electron_eta",hist_electronEta);
  //electron phi
  TH1F * hist_electronPhi = new TH1F("Electron_phi","Electron phi",50,-3.4,3.4);
  hist_electronPhi->SetMinimum(0);
  electronPhi_ = ibooker.book1D("Electron_phi",hist_electronPhi);

  //caloMET phi
  TH1F * hist_caloMetPhi = new TH1F("CaloMET_phi","CaloMET phi",50,-3.4,3.4);
  hist_caloMetPhi->SetMinimum(0);
  caloMetPhi_ = ibooker.book1D("CaloMET_phi",hist_caloMetPhi);

  //PFMET phi
  TH1F * hist_PFMetPhi = new TH1F("PFMET_phi","PFMET phi",50,-3.4,3.4);
  hist_PFMetPhi->SetMinimum(0);
  pfMetPhi_ = ibooker.book1D("PFMET_phi",hist_PFMetPhi);

  //bJet phi
  TH1F * hist_bJetPhi = new TH1F("bJet_phi","b-Jet phi",50,-3.4,3.4);
  hist_bJetPhi->SetMinimum(0);
  bJetPhi_ = ibooker.book1D("bJet_phi",hist_bJetPhi);
  //bJet eta
  TH1F * hist_bJetEta = new TH1F("bJet_eta","b-Jet eta",50,0,3);
  hist_bJetEta->SetMinimum(0);
  bJetEta_ = ibooker.book1D("bJet_eta",hist_bJetEta);

}

void HLTWorkspace::fillPlots(int evtNum, string pathName, edm::Handle<trigger::TriggerEvent> aodTriggerEvent)
{
  if (debugPrint) std::cout << "Inside fillPlots( " << evtNum << " , " << pathName << " ) " << std::endl;

  const TriggerObjectCollection objects = aodTriggerEvent->getObjects();
  
  edm::InputTag moduleFilter(lookupFilter[pathName],"","TEST");
  unsigned int moduleFilterIndex = aodTriggerEvent->filterIndex(moduleFilter);
  const Keys &keys = aodTriggerEvent->filterKeys( moduleFilterIndex );

  ////////////////////////////////
  ///
  /// single-object plots
  ///
  ////////////////////////////////  
  
  //alphaT
  if (pathName == "HLT_PFHT200_DiPFJet90_PFAlphaT0p57")
    {
      std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>> alphaT_jets;
      for (const auto & key : keys)
  	{
  	  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> JetLVec(objects[key].pt(),objects[key].eta(),objects[key].phi(),objects[key].mass());
  	  alphaT_jets.push_back(JetLVec);
  	}

      float alphaT = AlphaT(alphaT_jets,false).value(); 
      alphaT_->Fill(alphaT);
    }

  //photon pt + eta
  else if (pathName == "HLT_Photon30_R9Id90_HE10_IsoM")
    {
      for (const auto & key : keys) 
	{
	  photonPt_->Fill(objects[key].pt());
	  photonEta_->Fill(objects[key].eta());
	  photonPhi_->Fill(objects[key].phi());
	}
    }
  
  //muon pt + eta
  else if (pathName == "HLT_IsoMu27")
    {
      for (const auto & key : keys) 
  	{
  	  muonPt_->Fill(objects[key].pt());
  	  muonEta_->Fill(objects[key].eta());
  	  muonPhi_->Fill(objects[key].phi());
      	}
    }
  
  //electron pt + eta
  else if (pathName == "HLT_Ele27_eta2p1_WP75_Gsf")
    {
      for (const auto & key : keys) 
  	{
  	  electronPt_->Fill(objects[key].pt());
  	  electronEta_->Fill(objects[key].eta());
  	  electronPhi_->Fill(objects[key].phi());
  	}
    }
  
  //jet pt
  else if (pathName == "HLT_PFJet200")
    {
      for (const auto & key : keys) jetPt_->Fill(objects[key].pt());
    }

  //tau pt
  else if (pathName == "HLT_DoubleMediumIsoPFTau40_Trk1_eta2p1_Reg")
    {
      for (const auto & key : keys) tauPt_->Fill(objects[key].pt());
    }

  //caloMET pt+phi
  else if (pathName == "HLT_MET75_IsoTrk50")
    {
      for (const auto & key : keys)
  	{
  	  caloMetPt_->Fill(objects[key].pt());
  	  caloMetPhi_->Fill(objects[key].phi());
  	}
    }

  //caloHT pt
  else if (pathName == "HLT_HT650_DisplacedDijet80_Inclusive")
    {
      for (const auto & key : keys) caloHtPt_->Fill(objects[key].pt());
    }
  
  //PFHT pt
  else if (pathName == "HLT_PFHT750_4Jet")
    {
      for (const auto & key : keys) pfHtPt_->Fill(objects[key].pt());
    }

  //PFMET pt + phi
  else if (pathName == "HLT_PFMET120_PFMHT120_IDTight")
    {
      for (const auto & key : keys)
  	{
  	  pfMetPt_->Fill(objects[key].pt());
  	  pfMetPhi_->Fill(objects[key].phi());
  	}
    }

  // bjet eta + phi
  else if (pathName == "HLT_PFMET120_NoiseCleaned_BTagCSV07" || pathName == "HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq500")
    {
       for (const auto & key : keys)
  	{
   	  bJetEta_->Fill(objects[key].eta());
   	  bJetPhi_->Fill(objects[key].phi());
  	}
    }
  
  // ////////////////////////////////
  // ///
  // /// double-object plots
  // ///
  // ////////////////////////////////
  
  //double muon low mass 
  else if (pathName == "HLT_DoubleMu4_3_Jpsi_Displaced")
    {
      const double mu_mass(.105658);
      unsigned int kCnt0 = 0;  
      for (const auto & key0: keys)
  	{
  	  unsigned int kCnt1 = 0;
  	  for (const auto & key1: keys)
  	    {
  	      if (key0 != key1 && kCnt1 > kCnt0) // avoid filling hists with same objs && avoid double counting separate objs
  		{
  		  if (abs(objects[key0].id()) == 13 && abs(objects[key1].id()) == 13 && (objects[key0].id()+objects[key1].id()==0))  // check muon id and dimuon charge
  		    {
  		      TLorentzVector mu1, mu2, dimu;
  		      mu1.SetPtEtaPhiM(objects[key0].pt(), objects[key0].eta(), objects[key0].phi(), mu_mass);
  		      mu2.SetPtEtaPhiM(objects[key1].pt(), objects[key1].eta(), objects[key1].phi(), mu_mass);
  		      dimu = mu1+mu2;
  		      diMuonLowMass_->Fill(dimu.M());
  		    }
  		}
  	      kCnt1 +=1;
  	    }
  	  kCnt0 +=1;
  	}
    } //end double object plot
  
}

double HLTWorkspace::get_wall_time()
{
  struct timeval time;
  if (gettimeofday(&time,NULL)) return 0;
  return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

// ------------ method called when starting to processes a luminosity block  ------------
/*
void
HLTWorkspace::beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method called when ending the processing of a luminosity block  ------------
/*
void
HLTWorkspace::endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&)
{
}
*/

// ------------ method fills 'descriptions' with the allowed parameters for the module  ------------
// void
// HLTWorkspace::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
//   //The following says we do not know what parameters are allowed so do no validation
//   // Please change this to state exactly what you do use, even if it is no parameters
//   edm::ParameterSetDescription desc;
//   desc.setUnknown();
//   descriptions.addDefault(desc);
// }

//define this as a plug-in
DEFINE_FWK_MODULE(HLTWorkspace);
