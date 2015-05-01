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

// user include files
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/MonitorElement.h"

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
#include "HLTrigger/JetMET/interface/AlphaT.h"

#include "TMath.h"

#include "TStyle.h"

#include "TLorentzVector.h"

//
// class declaration
//

using namespace edm;
using namespace trigger;
using std::vector;
using std::string;
using std::map;

class HLTWorkspace : public edm::EDAnalyzer {
   public:
      explicit HLTWorkspace(const edm::ParameterSet&);
      ~HLTWorkspace();

  //      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() override;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) override;
      virtual void endJob() override;

      virtual void beginRun(edm::Run const&, edm::EventSetup const&) override;
      virtual void endRun(edm::Run const&, edm::EventSetup const&) override;
      //virtual void beginLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      //virtual void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
      virtual void bookPlots();

  virtual void fillPlots(int, string, edm::Handle<trigger::TriggerEvent>);
      // ----------member data ---------------------------

  bool debugPrint;

  string topDirectoryName;
  string mainShifterFolder;
  string backupFolder;

  DQMStore * dbe;

  HLTConfigProvider hltConfig_;

  map<string, unsigned int> lookupIndex;
  map<string, string> lookupFilter;

  vector<string> quickCollectionPaths;

  //set Token(-s)
  edm::EDGetTokenT<edm::TriggerResults> triggerResultsToken_;
  edm::EDGetTokenT<trigger::TriggerEvent> aodTriggerToken_;
  edm::EDGetTokenT<LumiScalersCollection> lumiScalersToken_;
  //  edm::EDGetTokenT<reco::JetTagCollection> csvTagToken_;

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

  topDirectoryName = TString("HLT/Workspaces");
  mainShifterFolder = TString(topDirectoryName+"/MainShifter");
  backupFolder = TString(topDirectoryName+"/Backup");

  //set Token(s)
  triggerResultsToken_ = consumes<edm::TriggerResults>(InputTag("TriggerResults","", "TEST"));
  aodTriggerToken_ = consumes<trigger::TriggerEvent>(InputTag("hltTriggerSummaryAOD", "", "TEST"));
  lumiScalersToken_ = consumes<LumiScalersCollection>(InputTag("hltScalersRawToDigi","",""));
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

}


// ------------ method called once each job just before starting event loop  ------------
void
HLTWorkspace::beginJob()
{
  if (debugPrint) std::cout << "Calling beginJob. " << std::endl;
  dbe = Service<DQMStore>().operator->();
  if (dbe) dbe->setCurrentFolder(topDirectoryName.c_str());
}

// ------------ method called once each job just after ending the event loop  ------------
void
HLTWorkspace::endJob()
{
}

// ------------ method called when starting to processes a run  ------------
void
HLTWorkspace::beginRun(edm::Run const& iRun, edm::EventSetup const& iSetup)
{
  if (debugPrint) std::cout << "Calling beginRun. " << std::endl;
  bool changed = true;
  if (hltConfig_.init(iRun, iSetup, "TEST", changed))
    {
      if (debugPrint) std::cout << "Extracting HLTconfig. " << std::endl;
    }

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

  bookPlots();

}

// ------------ method called when ending the processing of a run  ------------

void
HLTWorkspace::endRun(edm::Run const&, edm::EventSetup const&)
{
  if (debugPrint) std::cout << "Calling endRun. " << std::endl;
}

void HLTWorkspace::bookPlots()
{

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
  
  

  ////////////////////////////////
  ///
  /// Main shifter workspace plots
  ///
  ////////////////////////////////
  dbe->setCurrentFolder(mainShifterFolder);

  //photon pt
  TH1F * hist_photonPt = new TH1F("Photon_pT","Photon pT",50,0,100);
  hist_photonPt->SetMinimum(0);
  dbe->book1D("Photon_pT",hist_photonPt);

  //muon pt
  TH1F * hist_muonPt = new TH1F("Muon_pT","Muon pT",75,0,150);
  hist_muonPt->SetMinimum(0);
  dbe->book1D("Muon_pT",hist_muonPt);

  //electron pt
  TH1F * hist_electronPt = new TH1F("Electron_pT","Electron pT",75,0,150);
  hist_electronPt->SetMinimum(0);
  dbe->book1D("Electron_pT",hist_electronPt);

  //jet pt
  TH1F * hist_jetPt = new TH1F("Jet_pT","Jet pT",75,150,550);
  hist_jetPt->SetMinimum(0);
  dbe->book1D("Jet_pT",hist_jetPt);

  //dimuon low mass
  TH1F * hist_dimuonLowMass = new TH1F("Dimuon_LowMass","Dimuon Low Mass",100,2.5,3.5);
  hist_dimuonLowMass->SetMinimum(0);
  dbe->book1D("Dimuon_LowMass",hist_dimuonLowMass);

  //alphaT
  TH1F * hist_alphaT = new TH1F("alphaT","alphaT",30,0,5);
  hist_alphaT->SetMinimum(0);
  dbe->book1D("AlphaT",hist_alphaT);

  //caloMET pt
  TH1F * hist_caloMetPt = new TH1F("CaloMET_pT","CaloMET pT",75,0,150);
  hist_caloMetPt->SetMinimum(0);
  dbe->book1D("CaloMET_pT",hist_caloMetPt);

  ////////////////////////////////
  ///
  /// Backup workspace plots
  ///
  ////////////////////////////////
  dbe->setCurrentFolder(backupFolder);

  //photon eta
  TH1F * hist_photonEta = new TH1F("Photon_eta","Photon eta",50,0,3);
  hist_photonEta->SetMinimum(0);
  dbe->book1D("Photon_eta",hist_photonEta);
  //photon phi
  TH1F * hist_photonPhi = new TH1F("Photon_phi","Photon phi",50,-3.4,3.4);
  hist_photonPhi->SetMinimum(0);
  dbe->book1D("Photon_phi",hist_photonPhi);

  //muon eta
  TH1F * hist_muonEta = new TH1F("Muon_eta","Muon eta",50,0,3);
  hist_muonEta->SetMinimum(0);
  dbe->book1D("Muon_eta",hist_muonEta);
  //muon phi
  TH1F * hist_muonPhi = new TH1F("Muon_phi","Muon phi",50,-3.4,3.4);
  hist_muonPhi->SetMinimum(0);
  dbe->book1D("Muon_phi",hist_muonPhi);


  //electron eta
  TH1F * hist_electronEta = new TH1F("Electron_eta","Electron eta",50,0,3);
  hist_electronEta->SetMinimum(0);
  dbe->book1D("Electron_eta",hist_electronEta);
  //electron phi
  TH1F * hist_electronPhi = new TH1F("Electron_phi","Electron phi",50,-3.4,3.4);
  hist_electronPhi->SetMinimum(0);
  dbe->book1D("Electron_phi",hist_electronPhi);

}

void HLTWorkspace::fillPlots(int evtNum, string pathName, edm::Handle<trigger::TriggerEvent> aodTriggerEvent)
{
  if (debugPrint) std::cout << "Inside fillPlots( " << evtNum << " , " << pathName << " ) " << std::endl;

  const TriggerObjectCollection objects = aodTriggerEvent->getObjects();
  
  InputTag moduleFilter(lookupFilter[pathName],"","TEST");
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
      string fullPathAlphaT = mainShifterFolder+"/AlphaT";
      MonitorElement * ME_alphaT = dbe->get(fullPathAlphaT);
      TH1F * hist_alphaT = ME_alphaT->getTH1F();
      std::vector<ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>>> alphaT_jets;

      for (const auto & key : keys)
	{
	  ROOT::Math::LorentzVector<ROOT::Math::PtEtaPhiM4D<double>> JetLVec(objects[key].pt(),objects[key].eta(),objects[key].phi(),objects[key].mass());
	  alphaT_jets.push_back(JetLVec);
	}

      float alphaT = AlphaT(alphaT_jets,false).value(); 
      hist_alphaT->Fill(alphaT);
    }

  //photon pt + eta
  else if (pathName == "HLT_Photon30_R9Id90_HE10_IsoM")
    {
      //photon pt
      string fullPathPhotonPt = mainShifterFolder+"/Photon_pT";
      MonitorElement * ME_photonPt = dbe->get(fullPathPhotonPt);
      TH1F * hist_photonPt = ME_photonPt->getTH1F();
      //photon eta
      string fullPathPhotonEta = backupFolder+"/Photon_eta";
      MonitorElement * ME_photonEta = dbe->get(fullPathPhotonEta);
      TH1F * hist_photonEta = ME_photonEta->getTH1F();
      //photon phi
      string fullPathPhotonPhi = backupFolder+"/Photon_phi";
      MonitorElement * ME_photonPhi = dbe->get(fullPathPhotonPhi);
      TH1F * hist_photonPhi = ME_photonPhi->getTH1F();
    
      for (const auto & key : keys) 
	{
	  hist_photonPt->Fill(objects[key].pt());
	  hist_photonEta->Fill(objects[key].eta());
	  hist_photonPhi->Fill(objects[key].phi());
	}
    }
  
  //muon pt + eta
  else if (pathName == "HLT_IsoMu27")
    {
      //muon pt
      string fullPathMuonPt = mainShifterFolder+"/Muon_pT";
      MonitorElement * ME_muonPt = dbe->get(fullPathMuonPt);
      TH1F * hist_muonPt = ME_muonPt->getTH1F();
      //muon eta
      string fullPathMuonEta = backupFolder+"/Muon_eta";
      MonitorElement * ME_muonEta = dbe->get(fullPathMuonEta);
      TH1F * hist_muonEta = ME_muonEta->getTH1F();
      //muon phi
      string fullPathMuonPhi = backupFolder+"/Muon_phi";
      MonitorElement * ME_muonPhi = dbe->get(fullPathMuonPhi);
      TH1F * hist_muonPhi = ME_muonPhi->getTH1F();
      
      for (const auto & key : keys) 
	{
	  hist_muonPt->Fill(objects[key].pt());
	  hist_muonEta->Fill(objects[key].eta());
	  hist_muonPhi->Fill(objects[key].phi());
      	}
    }
  
  //electron pt + eta
  else if (pathName == "HLT_Ele27_eta2p1_WP75_Gsf")
    {
      //electron pt
      string fullPathElectronPt = mainShifterFolder+"/Electron_pT";
      MonitorElement * ME_electronPt = dbe->get(fullPathElectronPt);
      TH1F * hist_electronPt = ME_electronPt->getTH1F();
      //electron eta
      string fullPathElectronEta = backupFolder+"/Electron_eta";
      MonitorElement * ME_electronEta = dbe->get(fullPathElectronEta);
      TH1F * hist_electronEta = ME_electronEta->getTH1F();
      //electron phi
      string fullPathElectronPhi = backupFolder+"/Electron_phi";
      MonitorElement * ME_electronPhi = dbe->get(fullPathElectronPhi);
      TH1F * hist_electronPhi = ME_electronPhi->getTH1F();

      for (const auto & key : keys) 
	{
	  hist_electronPt->Fill(objects[key].pt());
	  hist_electronEta->Fill(objects[key].eta());
	  hist_electronPhi->Fill(objects[key].phi());
	}
    }
  
  //jet pt
  else if (pathName == "HLT_PFJet200")
    {
      //jet pt
      string fullPathJetPt = mainShifterFolder+"/Jet_pT";
      MonitorElement * ME_jetPt = dbe->get(fullPathJetPt);
      TH1F * hist_jetPt = ME_jetPt->getTH1F();
      for (const auto & key : keys) hist_jetPt->Fill(objects[key].pt());
    }

  //caloMET pt
  else if (pathName == "HLT_MET75_IsoTrk50")
    {
      // pt
      string fullPathCaloMetPt = mainShifterFolder+"/CaloMET_pT";
      MonitorElement * ME_caloMetPt = dbe->get(fullPathCaloMetPt);
      TH1F * hist_caloMetPt = ME_caloMetPt->getTH1F();
      for (const auto & key : keys) hist_caloMetPt->Fill(objects[key].pt());
    }

  
  
 //CSV
  // else if (pathName == "HLT_QuadPFJet_SingleBTagCSV_VBF_Mqq240" || pathName == "HLT_PFMET120_NoiseCleaned_BTagCSV07")
  //   {
  //     string fullPathBjetCsv = mainShifterFolder+"";
  //     //ME;
  //     //      TH1F*;
  //     double csvTag = lookupCsv[trgObj.pt()];
  //     hist->Fill(csvTag);
  //   }

  ////////////////////////////////
  ///
  /// double-object plots
  ///
  ////////////////////////////////
  
  //double muon low mass
  else if (pathName == "HLT_DoubleMu4_3_Jpsi_Displaced")
    {
      string fullPathDimuonLowMass = mainShifterFolder+"/Dimuon_LowMass";
      MonitorElement * ME_dimuonLowMass = dbe->get(fullPathDimuonLowMass);
      TH1F * hist_dimuonLowMass = ME_dimuonLowMass->getTH1F();
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
		      hist_dimuonLowMass->Fill(dimu.M());
		    }
		}
	      kCnt1 +=1;
	    }
	  kCnt0 +=1;
	}
    } //end double object plot
  
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
