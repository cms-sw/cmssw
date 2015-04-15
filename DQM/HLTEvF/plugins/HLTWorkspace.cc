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

  //dimuon low mass
  TH1F * hist_dimuonLowMass = new TH1F("Dimuon_LowMass","Dimuon Low Mass",100,2.5,3.5);
  hist_dimuonLowMass->SetMinimum(0);
  dbe->book1D("Dimuon_LowMass",hist_dimuonLowMass);

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

  //muon eta
  TH1F * hist_muonEta = new TH1F("Muon_eta","Muon eta",50,0,3);
  hist_muonEta->SetMinimum(0);
  dbe->book1D("Muon_eta",hist_muonEta);

  //electron eta
  TH1F * hist_electronEta = new TH1F("Electron_eta","Electron eta",50,0,3);
  hist_electronEta->SetMinimum(0);
  dbe->book1D("Electron_eta",hist_electronEta);

}

void HLTWorkspace::fillPlots(int evtNum, string pathName, edm::Handle<trigger::TriggerEvent> aodTriggerEvent)
{
  if (debugPrint) std::cout << "Inside fillPlots( " << evtNum << " , " << pathName << " ) " << std::endl;

   const TriggerObjectCollection objects = aodTriggerEvent->getObjects();
  InputTag moduleFilter(lookupFilter[pathName],"","TEST");
  unsigned int moduleFilterIndex = aodTriggerEvent->filterIndex(moduleFilter);
  const Keys &keys = aodTriggerEvent->filterKeys( moduleFilterIndex );

  unsigned int kCnt1 = 0;
  for (const auto & key1 : keys)
    {
      TriggerObject trgObj1 = objects[key1];

      ////////////////////////////////
      ///
      /// single-object plots
      ///
      ////////////////////////////////

      //photon pt + eta
      if (pathName == "HLT_Photon30_R9Id90_HE10_IsoM")
	{
	  //photon pt
	  string fullPathPhotonPt = mainShifterFolder+"/Photon_pT";
	  MonitorElement * ME_photonPt = dbe->get(fullPathPhotonPt);
	  TH1F * hist_photonPt = ME_photonPt->getTH1F();
	  hist_photonPt->Fill(trgObj1.pt());

	  //photon eta
	  string fullPathPhotonEta = backupFolder+"/Photon_eta";
	  MonitorElement * ME_photonEta = dbe->get(fullPathPhotonEta);
	  TH1F * hist_photonEta = ME_photonEta->getTH1F();
	  hist_photonEta->Fill(trgObj1.eta());
	}

      //muon pt + eta
      else if (pathName == "HLT_IsoMu27")
	{
	  //muon pt
	  string fullPathMuonPt = mainShifterFolder+"/Muon_pT";
	  MonitorElement * ME_muonPt = dbe->get(fullPathMuonPt);
	  TH1F * hist_muonPt = ME_muonPt->getTH1F();
	  hist_muonPt->Fill(trgObj1.pt());

	  //muon eta
	  string fullPathMuonEta = backupFolder+"/Muon_eta";
	  MonitorElement * ME_muonEta = dbe->get(fullPathMuonEta);
	  TH1F * hist_muonEta = ME_muonEta->getTH1F();
	  hist_muonEta->Fill(trgObj1.eta());
	}

      //electron pt + eta
      else if (pathName == "HLT_Ele27_eta2p1_WP75_Gsf")
	{
	  //electron pt
	  string fullPathElectronPt = mainShifterFolder+"/Electron_pT";
	  MonitorElement * ME_electronPt = dbe->get(fullPathElectronPt);
	  TH1F * hist_electronPt = ME_electronPt->getTH1F();
	  hist_electronPt->Fill(trgObj1.pt());

	  //electron eta
	  string fullPathElectronEta = backupFolder+"/Electron_eta";
	  MonitorElement * ME_electronEta = dbe->get(fullPathElectronEta);
	  TH1F * hist_electronEta = ME_electronEta->getTH1F();
	  hist_electronEta->Fill(trgObj1.eta());
	}

      //start second for loop for double-object plots
      unsigned int kCnt2 = 0;
      for (const auto & key2 : keys)
      	{
      	  if (key1 != key2 && kCnt2 > kCnt1) // avoid filling hists with same objs && avoid double counting separate objs
      	    {
      	      TriggerObject trgObj2 = objects[key2];

      	      ////////////////////////////////
      	      ///
      	      /// double-object plots
      	      ///
      	      ////////////////////////////////

              //double muon low mass
              if (pathName == "HLT_DoubleMu4_3_Jpsi_Displaced")
                {
                  if (abs(trgObj1.id()) == 13 && abs(trgObj2.id()) == 13 && (trgObj1.id()+trgObj2.id()==0)) { // check muon id and dimuon charge
                    //dimuon low mass
                    string fullPathDimuonLowMass = mainShifterFolder+"/Dimuon_LowMass";
                    MonitorElement * ME_dimuonLowMass = dbe->get(fullPathDimuonLowMass);
                    TH1F * hist_dimuonLowMass = ME_dimuonLowMass->getTH1F();
                    const double mu_mass(.105658);
                    TLorentzVector mu1, mu2, dimu;
                    mu1.SetPtEtaPhiM(trgObj1.pt(), trgObj1.eta(), trgObj1.phi(), mu_mass);
                    mu2.SetPtEtaPhiM(trgObj2.pt(), trgObj2.eta(), trgObj2.phi(), mu_mass);
                    dimu = mu1+mu2;
                    hist_dimuonLowMass->Fill(dimu.M());
                  }
                }

      	    }
      	  kCnt2 +=1;
      	} //end second for loop over trig objs
      kCnt1 +=1;
    } //end first for loop over trig objs
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
