/*
 * \file DQMHcalPhiSymAlCaReco.cc
 *
 * \author Olga Kodolova
 *        
 *
 *
 * Description: Monitoring of Phi Symmetry Calibration Stream  
*/

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/Service.h"

// DQM include files

#include "DQMServices/Core/interface/MonitorElement.h"

// work on collections

#include "DataFormats/CaloTowers/interface/CaloTowerDetId.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"


// #include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
// #include "Geometry/CaloGeometry/interface/CaloCellGeometry.h"
// #include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMOffline/CalibCalo/src/DQMHcalDiJetsAlCaReco.h"

using namespace std;
using namespace edm;
using namespace reco;

// ******************************************
// constructors
// *****************************************

DQMHcalDiJetsAlCaReco::DQMHcalDiJetsAlCaReco( const edm::ParameterSet& iConfig ) :
eventCounter_(0)
{
  dbe_ = Service<DQMStore>().operator->();
//
// Input from configurator file 
//
  folderName_ = iConfig.getUntrackedParameter<string>("FolderName","ALCAStreamHcalDiJets");


  jets_= consumes<CaloJetCollection>(iConfig.getParameter<edm::InputTag>("jetsInput"));
  ec_= consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("ecInput"));
  hbhe_= consumes<HBHERecHitCollection>(iConfig.getParameter<edm::InputTag>("hbheInput"));
  ho_= consumes<HORecHitCollection>(iConfig.getParameter<edm::InputTag>("hoInput"));
  hf_= consumes<HFRecHitCollection>(iConfig.getParameter<edm::InputTag>("hfInput"));
  
  saveToFile_= iConfig.getUntrackedParameter<bool>("SaveToFile",false);
  fileName_=  iConfig.getUntrackedParameter<string>("FileName","MonitorAlCaHcalDiJets.root");
    
}

DQMHcalDiJetsAlCaReco::~DQMHcalDiJetsAlCaReco()
{}

//--------------------------------------------------------
void DQMHcalDiJetsAlCaReco::beginJob(){
   

  // create and cd into new folder
  dbe_->setCurrentFolder(folderName_);

  // book some histograms 1D
  hiDistrRecHitEnergyEBEE_ = dbe_->book1D("RecHitEnergyEBEE", "the number of hits inside jets", 100, 0, 800);
  hiDistrRecHitEnergyEBEE_->setAxisTitle("E, GeV", 1);
  hiDistrRecHitEnergyEBEE_->setAxisTitle("# rechits", 2);

  hiDistrRecHitEnergyHBHE_ = dbe_->book1D("RecHitEnergyHBHE", "the number of hits inside jets", 100,0, 800);
  hiDistrRecHitEnergyHBHE_->setAxisTitle("E, GeV", 1);
  hiDistrRecHitEnergyHBHE_->setAxisTitle("# rechits", 2);

  hiDistrRecHitEnergyHF_ = dbe_->book1D("RecHitEnergyHF", "the number of hits inside jets", 150,0, 1500); 
  hiDistrRecHitEnergyHF_->setAxisTitle("E, GeV", 1);
  hiDistrRecHitEnergyHF_->setAxisTitle("# rechits", 2);

  hiDistrRecHitEnergyHO_ = dbe_->book1D("RecHitEnergyHO", "the number of hits inside jets", 100,0, 100); 
  hiDistrRecHitEnergyHO_->setAxisTitle("E, GeV", 1);
  hiDistrRecHitEnergyHO_->setAxisTitle("# rechits", 2);

  hiDistrProbeJetEnergy_ = dbe_->book1D("ProbeJetEnergy", "the energy of probe jets", 250,0, 2500); 
  hiDistrProbeJetEnergy_->setAxisTitle("E, GeV", 1);
  hiDistrProbeJetEnergy_->setAxisTitle("# jets", 2);

  hiDistrProbeJetEta_ = dbe_->book1D("ProbeJetEta", "the number of probe jets", 100, -5., 5.); 
  hiDistrProbeJetEta_->setAxisTitle("#eta", 1);
  hiDistrProbeJetEta_->setAxisTitle("# jets", 2);

  hiDistrProbeJetPhi_ = dbe_->book1D("ProbeJetPhi", "the number of probe jets", 50, -3.14, 3.14); 
  hiDistrProbeJetPhi_->setAxisTitle("#phi", 1);
  hiDistrProbeJetPhi_->setAxisTitle("# jets", 2);

  hiDistrTagJetEnergy_ = dbe_->book1D("TagJetEnergy", "the energy of tsg jets", 250,0, 2500); 
  hiDistrTagJetEnergy_->setAxisTitle("E, GeV", 1);
  hiDistrTagJetEnergy_->setAxisTitle("# jets", 2);

  hiDistrTagJetEta_ = dbe_->book1D("TagJetEta", "the number of  tag jets", 100, -5., 5.); 
  hiDistrTagJetEta_->setAxisTitle("#eta", 1);
  hiDistrTagJetEta_->setAxisTitle("# jets", 2);

  hiDistrTagJetPhi_ = dbe_->book1D("TagJetPhi", "the number of tag jets", 50, -3.14, 3.14); 
  hiDistrTagJetPhi_->setAxisTitle("#phi", 1);
  hiDistrTagJetPhi_->setAxisTitle("# jets", 2);

  hiDistrEtThirdJet_ = dbe_->book1D("EtThirdJet", "Et of the third jet", 90, 0, 90); 


//==================================================================================


}

//--------------------------------------------------------
void DQMHcalDiJetsAlCaReco::beginRun(const edm::Run& r, const EventSetup& context) {

}

//--------------------------------------------------------
void DQMHcalDiJetsAlCaReco::beginLuminosityBlock(const LuminosityBlock& lumiSeg, 
     const EventSetup& context) {
  
}

//-------------------------------------------------------------

void DQMHcalDiJetsAlCaReco::analyze(const Event& iEvent, 
			       const EventSetup& iSetup ){  
 

   eventCounter_++;

   CaloJet jet1, jet2, jet3;
   Float_t etVetoJet; 

   edm::Handle<CaloJetCollection> jets;
   iEvent.getByToken(jets_,jets);
   
  if(!jets.isValid()){
    LogDebug("") << "DQMHcalDiJetsAlCaReco: Error! can't getjet product!" << std::endl;
    return ;
  }
   
   
   
   if(jets->size()>1){
    jet1 = (*jets)[0];
    jet2 = (*jets)[1];
     if(fabs(jet1.eta())>fabs(jet2.eta())){
       CaloJet jet = jet1;
       jet1 = jet2;
       jet2 = jet;
     }
     //     if(fabs(jet1.eta())>eta_1 || (fabs(jet2.eta())-jet_R) < eta_2){ return;}
   } else {return;}

   hiDistrTagJetEnergy_->Fill(jet1.energy());
   hiDistrTagJetEta_->Fill(jet1.eta());
   hiDistrTagJetPhi_->Fill(jet1.phi());

   hiDistrProbeJetEnergy_->Fill(jet2.energy());
   hiDistrProbeJetEta_->Fill(jet2.eta());
   hiDistrProbeJetPhi_->Fill(jet2.phi());

   if(jets->size()>2){
     jet3 = (*jets)[2];
     etVetoJet = jet3.et();
     hiDistrEtThirdJet_->Fill(etVetoJet);
   } else { etVetoJet = 0.; hiDistrEtThirdJet_->Fill(etVetoJet); }



      Handle<EcalRecHitCollection> ec;
      iEvent.getByToken(ec_,ec);
      
  if(!ec.isValid()){
    LogDebug("") << "DQMHcalDiJetsAlCaReco: Error! can't get ec product!" << std::endl;
    return ;
  }
      
      
      for(EcalRecHitCollection::const_iterator ecItr = (*ec).begin();
                                                ecItr != (*ec).end(); ++ecItr)
      {
        hiDistrRecHitEnergyEBEE_->Fill(ecItr->energy()); 
      }




      Handle<HBHERecHitCollection> hbhe;
      iEvent.getByToken(hbhe_, hbhe);

  if(!hbhe.isValid()){
    LogDebug("") << "DQMHcalDiJetsAlCaReco: Error! can't get hbhe product!" << std::endl;
    return ;
  }


      for(HBHERecHitCollection::const_iterator hbheItr=hbhe->begin();
                                                 hbheItr!=hbhe->end(); hbheItr++)
      {
	hiDistrRecHitEnergyHBHE_->Fill(hbheItr->energy()); 
      }

   
      Handle<HORecHitCollection> ho;
      iEvent.getByToken(ho_, ho);

  if(!ho.isValid()){
    LogDebug("") << "DQMHcalDiJetsAlCaReco: Error! can't get ho product!" << std::endl;
    return ;
  }


      for(HORecHitCollection::const_iterator hoItr=ho->begin();
                                               hoItr!=ho->end(); hoItr++)
      {
         hiDistrRecHitEnergyHO_->Fill(hoItr->energy());

      }




      Handle<HFRecHitCollection> hf;
      iEvent.getByToken(hf_, hf);

  if(!hf.isValid()){
    LogDebug("") << "DQMHcalDiJetsAlCaReco: Error! can't get hf product!" << std::endl;
    return ;
  }


      for(HFRecHitCollection::const_iterator hfItr=hf->begin();
                                               hfItr!=hf->end(); hfItr++)
      {
	hiDistrRecHitEnergyHF_->Fill(hfItr->energy()); 
      }
	
} //analyze




//--------------------------------------------------------
void DQMHcalDiJetsAlCaReco::endLuminosityBlock(const LuminosityBlock& lumiSeg, 
                                          const EventSetup& context) {
}
//--------------------------------------------------------
void DQMHcalDiJetsAlCaReco::endRun(const Run& r, const EventSetup& context){

}
//--------------------------------------------------------
void DQMHcalDiJetsAlCaReco::endJob(){
  
  if (saveToFile_) {
     dbe_->save(fileName_);
  }
  
}


