// system include files
#include <memory>
#include <string>
#include <iostream>
#include <algorithm>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/EDAnalyzer.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Utilities/interface/EDGetToken.h"
//#include "Calibration/EcalAlCaRecoProducers/interface/AlCaElectronsProducer.h"
#include "DataFormats/EgammaCandidates/interface/SiStripElectron.h"
#include "DataFormats/EgammaCandidates/interface/ElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
#include "DataFormats/EgammaReco/interface/BasicCluster.h"


#include "TFile.h"
#include "TH1.h"
#include "TH2.h"

#include <Math/VectorUtil.h>

class AlCaElectronsTest : public edm::EDAnalyzer {
  public:
    explicit AlCaElectronsTest (const edm::ParameterSet&) ;
    ~AlCaElectronsTest () {}
     virtual void analyze (const edm::Event& iEvent, 
                           const edm::EventSetup& iSetup) ;
     virtual void beginJob() ;
     virtual void endJob () ;

  private:

     EcalRecHit getMaximum (const EcalRecHitCollection * recHits) ;
     void fillAroundBarrel (const EcalRecHitCollection * recHits, int eta, int phi) ;
     void fillAroundEndcap (const EcalRecHitCollection * recHits, int ics, int ips) ;


  private:

    edm::EDGetTokenT<EBRecHitCollection> m_barrelAlCa ;
    edm::EDGetTokenT<EERecHitCollection> m_endcapAlCa ;
    std::string   m_outputFileName ;            

    //! ECAL map
    TH2F * m_barrelGlobalCrystalsMap ;
    //! local map
    TH2F * m_barrelLocalCrystalsMap ;
    //! ECAL map
    TH2F * m_endcapGlobalCrystalsMap ;
    //! local map
    TH2F * m_endcapLocalCrystalsMap ;
    //! ECAL Energy
    TH2F * m_barrelGlobalCrystalsEnergy ;
    //! local Energy
    TH2F * m_barrelLocalCrystalsEnergy ;
    //! ECAL Energy
    TH2F * m_endcapGlobalCrystalsEnergy ;
    //! local Energy
    TH2F * m_endcapLocalCrystalsEnergy ;
    //! ECAL EnergyMap
    TH2F * m_barrelGlobalCrystalsEnergyMap ;
    //! ECAL EnergyMap
    TH2F * m_endcapGlobalCrystalsEnergyMap ;
} ;


// ----------------------------------------------------------------


AlCaElectronsTest::AlCaElectronsTest (const edm::ParameterSet& iConfig) :
  m_barrelAlCa (consumes<EBRecHitCollection>(iConfig.getParameter<edm::InputTag> ("alcaBarrelHitCollection"))) ,
  m_endcapAlCa (consumes<EERecHitCollection>(iConfig.getParameter<edm::InputTag> ("alcaEndcapHitCollection"))) ,
  m_outputFileName (iConfig.getUntrackedParameter<std::string>
                      ("HistOutFile",std::string ("AlCaElectronsTest.root"))) 
{}


// ----------------------------------------------------------------


void 
AlCaElectronsTest::beginJob()
{
  m_barrelGlobalCrystalsMap = new TH2F ("m_barrelGlobalCrystalsMap","m_barrelGlobalCrystalsMap",171,-85,86,360,0,360) ;
  m_barrelLocalCrystalsMap = new TH2F ("m_barrelLocalCrystalsMap","m_barrelLocalCrystalsMap",20,-10,10,20,-10,10) ;
  m_endcapGlobalCrystalsMap = new TH2F ("m_endcapGlobalCrystalsMap","m_endcapGlobalCrystalsMap",100,0,100,100,0,100) ;
  m_endcapLocalCrystalsMap = new TH2F ("m_endcapLocalCrystalsMap","m_endcapLocalCrystalsMap",20,-10,10,20,-10,10) ;
  m_barrelGlobalCrystalsEnergy = new TH2F ("m_barrelGlobalCrystalsEnergy","m_barrelGlobalCrystalsEnergy",171,-85,86,360,0,360) ;
  m_barrelLocalCrystalsEnergy = new TH2F ("m_barrelLocalCrystalsEnergy","m_barrelLocalCrystalsEnergy",20,-10,10,20,-10,10) ;
  m_endcapGlobalCrystalsEnergy = new TH2F ("m_endcapGlobalCrystalsEnergy","m_endcapGlobalCrystalsEnergy",100,0,100,100,0,100) ;
  m_endcapLocalCrystalsEnergy = new TH2F ("m_endcapLocalCrystalsEnergy","m_endcapLocalCrystalsEnergy",20,-10,10,20,-10,10) ;
  m_barrelGlobalCrystalsEnergyMap = new TH2F ("m_barrelGlobalCrystalsEnergyMap","m_barrelGlobalCrystalsEnergyMap",171,-85,86,360,0,360) ;
  m_endcapGlobalCrystalsEnergyMap = new TH2F ("m_endcapGlobalCrystalsEnergyMap","m_endcapGlobalCrystalsEnergyMap",100,0,100,100,0,100) ;
   return ;
}


// ----------------------------------------------------------------


void 
AlCaElectronsTest::endJob ()
{      
   TFile output (m_outputFileName.c_str (),"recreate") ;
   m_barrelGlobalCrystalsMap->Write () ;
   m_barrelLocalCrystalsMap->Write () ;
   m_endcapGlobalCrystalsMap->Write () ;
   m_endcapLocalCrystalsMap->Write () ;   
   m_barrelGlobalCrystalsEnergy->Write () ;
   m_barrelLocalCrystalsEnergy->Write () ;
   m_endcapGlobalCrystalsEnergy->Write () ;
   m_endcapLocalCrystalsEnergy->Write () ;   
   m_barrelGlobalCrystalsEnergyMap->Write () ;
   m_endcapGlobalCrystalsEnergyMap->Write () ;
   output.Close () ;
   //PG save root things
   return ;
}


// ----------------------------------------------------------------


void 
AlCaElectronsTest::analyze (const edm::Event& iEvent, 
                            const edm::EventSetup& iSetup)
{
  //FIXME replace with msg logger
  std::cout << "[AlCaElectronsTest] analysing event " 
            << iEvent.id () << std::endl ;

  //PG get the collections  
  // get Barrel RecHits
  edm::Handle<EBRecHitCollection> barrelRecHitsHandle ;
  iEvent.getByToken (m_barrelAlCa, barrelRecHitsHandle) ;
  if (!barrelRecHitsHandle.isValid()) {
      edm::EDConsumerBase::Labels labels;
      labelsForToken(m_barrelAlCa, labels);
      std::cerr << "[AlCaElectronsTest] caught std::exception "
                << " in rertieving " << labels.module
                << std::endl ;
      return ;
  } else {
      const EBRecHitCollection* barrelHitsCollection = barrelRecHitsHandle.product () ;
      //PG fill the histo with the maximum
      EcalRecHit barrelMax = getMaximum (barrelHitsCollection) ;
      EBDetId barrelMaxId (barrelMax.id ()) ; 
      m_barrelGlobalCrystalsMap->Fill (
          barrelMaxId.ieta () ,
          barrelMaxId.iphi () 
        ) ;
      m_barrelGlobalCrystalsEnergy->Fill (
          barrelMaxId.ieta () ,
          barrelMaxId.iphi () ,
          barrelMax.energy ()
        ) ;
      fillAroundBarrel (
          barrelHitsCollection, 
          barrelMaxId.ieta (), 
          barrelMaxId.iphi ()
        ) ;
  }
  
  // get Endcap RecHits
  edm::Handle<EERecHitCollection> endcapRecHitsHandle ;
  iEvent.getByToken (m_endcapAlCa,endcapRecHitsHandle) ;
  if (!endcapRecHitsHandle.isValid()) {
      edm::EDConsumerBase::Labels labels;
      labelsForToken(m_endcapAlCa, labels);
      std::cerr << "[AlCaElectronsTest] caught std::exception " 
                << " in rertieving " << labels.module
                << std::endl ;
      return ;
  } else {
      const EERecHitCollection* endcapHitsCollection = endcapRecHitsHandle.product () ;
      //PG fill the histo with the maximum
      EcalRecHit endcapMax = getMaximum (endcapHitsCollection) ;
      EEDetId endcapMaxId (endcapMax.id ()) ; 
      m_endcapGlobalCrystalsMap->Fill (
          endcapMaxId.ix () ,
          endcapMaxId.iy () 
        ) ;
      m_endcapGlobalCrystalsEnergy->Fill (
          endcapMaxId.ix () ,
          endcapMaxId.iy () ,
          endcapMax.energy ()
        ) ;
      fillAroundEndcap (
          endcapHitsCollection, 
          endcapMaxId.ix (), 
          endcapMaxId.iy ()
        ) ;
  }
}


// ----------------------------------------------------------------


EcalRecHit
AlCaElectronsTest::getMaximum (const EcalRecHitCollection * recHits) 
{
  double energy = 0. ;
  EcalRecHit max ;
  for (EcalRecHitCollection::const_iterator elem = recHits->begin () ;
       elem != recHits->end () ;
       ++elem)
    {
      if (elem->energy () > energy)
        {
          energy = elem->energy () ;
          max = *elem ;
        }        
    }   
  return max ;
}


// ----------------------------------------------------------------


void
AlCaElectronsTest::fillAroundBarrel (const EcalRecHitCollection * recHits, int eta, int phi)
{
  for (EcalRecHitCollection::const_iterator elem = recHits->begin () ;
       elem != recHits->end () ;
       ++elem)
    {
      EBDetId elementId = elem->id () ; 
      m_barrelLocalCrystalsMap->Fill (
        elementId.ieta () - eta ,
        elementId.iphi () - phi 
      ) ;
      m_barrelLocalCrystalsEnergy->Fill (
        elementId.ieta () - eta ,
        elementId.iphi () - phi ,
        elem->energy ()
      ) ;
     m_barrelGlobalCrystalsEnergyMap->Fill (
        elementId.ieta () ,
        elementId.iphi () ,
        elem->energy ()
      ) ;

    }   
  return ;
}


// ----------------------------------------------------------------


void
AlCaElectronsTest::fillAroundEndcap (const EcalRecHitCollection * recHits, int ics, int ips)
{
  for (EcalRecHitCollection::const_iterator elem = recHits->begin () ;
       elem != recHits->end () ;
       ++elem)
    {
      EEDetId elementId = elem->id () ; 
      m_endcapLocalCrystalsMap->Fill (
        elementId.ix () - ics ,
        elementId.iy () - ips 
      ) ;
      m_endcapLocalCrystalsEnergy->Fill (
        elementId.ix () - ics ,
        elementId.iy () - ips ,
        elem->energy ()
      ) ;
      m_endcapGlobalCrystalsEnergyMap->Fill (
        elementId.ix () ,
        elementId.iy () ,
        elem->energy ()
      ) ;
    }   
  return ;
}
