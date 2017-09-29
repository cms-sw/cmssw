// -*- C++ -*-
//
// Package:    EcalBadCalibFilter
// Class:      EcalBadCalibFilter
//
/**\class EcalBadCalibFilter EcalBadCalibFilter.cc
 
 Description: <one line class summary>
 Event filtering to remove events with anomalous energy intercalibrations in specific ECAL channels
*/
//
// Original Authors:  D. Petyt
//
 
 
// include files
 
#include <iostream>
 
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/global/EDFilter.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
 
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/EcalDetId/interface/EEDetId.h"
 
#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloEventSetup/interface/CaloTopologyRecord.h"
 
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "RecoCaloTools/Navigation/interface/CaloNavigator.h"
 
#include "TVector3.h"
 
using namespace std;
 
 
class EcalBadCalibFilter : public edm::global::EDFilter<> {
 
public:
 
  explicit EcalBadCalibFilter(const edm::ParameterSet & iConfig);
  ~EcalBadCalibFilter() override {}
 
private:
 
  // main filter function
 
 bool filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const override; 
 
 
  // input parameters
 
  // eb rechit collection (from AOD)
  const edm::EDGetTokenT<EcalRecHitCollection>  ebRHSrcToken_;  
  // ee rechit collection (from AOD)
  const edm::EDGetTokenT<EcalRecHitCollection>  eeRHSrcToken_;

  //config parameters (defining the cuts on the bad SCs)
  const double EBmin_;              // EB rechit et threshold
  const double EEmin_;              // EE rechit et threshold
 
  const std::vector<unsigned int> baddetEB_;    // DetIds of bad EB channels  
  const std::vector<unsigned int> baddetEE_;    // DetIds of bad EE channels
 
  const bool taggingMode_;
  const bool debug_;                // prints out debug info if set to true
 
};
 
// read the parameters from the config file
EcalBadCalibFilter::EcalBadCalibFilter(const edm::ParameterSet & iConfig)
  : ebRHSrcToken_     (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHitSource")))
  , eeRHSrcToken_     (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitSource")))
  , EBmin_          (iConfig.getParameter<double>("EBminet"))
  , EEmin_          (iConfig.getParameter<double>("EEminet"))
  , baddetEB_       (iConfig.getParameter<std::vector<unsigned int> >("baddetEB"))
  , baddetEE_       (iConfig.getParameter<std::vector<unsigned int> >("baddetEE"))
  , taggingMode_    (iConfig.getParameter<bool>("taggingMode"))
  , debug_          (iConfig.getParameter<bool>("debug"))
{
  produces<bool>();
}
 

bool EcalBadCalibFilter::filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {


  // load required collections
 
 
  // EB rechit collection
  edm::Handle<EcalRecHitCollection> ebRHs;
  iEvent.getByToken(eeRHSrcToken_, ebRHs);
 
  // EE rechit collection
  edm::Handle<EcalRecHitCollection> eeRHs;
  iEvent.getByToken(eeRHSrcToken_, eeRHs);
 
  // Calo Geometry - needed for computing E_t
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
 
  
  // by default the event is OK
  bool pass = true;
 
 
 
  // define energy variables and ix,iy,iz coordinates
 
 
  int ix,iy,iz;
  ix=0,iy=0,iz=0;
  float ene=0;
  float et=0;
 
 
 
  //EE:  loop over the list of bad DetIds (defined in the python file)
 
  //  std::cout << "Starting EE loop" << std::endl;


  for (std::vector<unsigned int>::const_iterator eeit = baddetEE_.begin(); eeit != baddetEE_.end(); ++eeit) {
 
    // std::cout << "In EE loop" << std::endl;
     
    EEDetId eedet(*eeit);
 
    // std::cout << "got detid" << std::endl;
 
    if (eedet.rawId()==0) continue;
 
    // std::cout << "Searching for DetId=" << eedet.rawId() << std::endl;
 
 
    // find rechit corresponding to this DetId
 
    EcalRecHitCollection::const_iterator eehit=eeRHs->find(eedet);
 
    // std::cout << "executed eeRHs->find()" << std::endl;
 
    if (eehit==eeRHs->end()) continue;
 
    // std::cout << "detid found=" << eehit->id().rawId() << std::endl;
 
    // if rechit not found, move to next DetId   
 
    if (eehit->id().rawId()==0 || eehit->id().rawId()!= eedet.rawId()) {
 
      //      std::cout << "Detid " << eedet.rawId() << " not found" << std::endl;
      continue;
    
    }
     
 
    //  std::cout << "**FOUND Detid " << eedet.rawId() << std::endl;
 
 
    // rechit has been found: obtain crystal coordinates, energy 
 
 
    ix=eedet.ix();
    iy=eedet.iy();
    iz=eedet.zside();
    ene=eehit->energy();
 
    // compute transverse energy
     
    //    std::cout << "accessing geometry" << std::endl;
 
    GlobalPoint posee=pG->getPosition(eedet);
    float pf=1.0/cosh(posee.eta());
    et=ene*pf;
     
    /*
      std::cout << "Detid=" << eedet.rawId() 
      << " ix=" << ix << " iy=" << iy<< " iz=" << iz 
      << " energy=" << ene << " et=" << et << std::endl;
    */
 
    // print some debug info
       
    if (debug_) {
      edm::LogInfo("EcalBadCalibFilter") << "DetId=" <<  eedet.rawId();
      edm::LogInfo("EcalBadCalibFilter") << "ix=" << ix << " iy=" << iy << " iz=" << iz;
      edm::LogInfo("EcalBadCalibFilter") << "Et=" << et << " thresh=" << EEmin_;
    }
       
       
    // if transverse energy is above threshold and channel has bad IC 

    if (et>EEmin_) {
      pass=false;
      std::cout << "DUMP EVENT" << std::endl;
    }
     
    //    std::cout << " end loop - going to next" << std::endl;
 
  }
 
 
  //  std::cout << "END of EE loop" << std::endl;
   
 
  // print the decision if event is bad
  if (pass==false && debug_) edm::LogInfo("EcalBadCalibFilter") << "REJECT EVENT!!!";
   
  // std::cout << "putting decision into event" << std::endl;  
 
  iEvent.put(std::make_unique<bool>(pass));
 
  // return the decision
 
  // std::cout << "Returning the decision..." << std::endl;
 
  return taggingMode_ || pass;
}
 
 
#include "FWCore/Framework/interface/MakerMacros.h"
 
DEFINE_FWK_MODULE(EcalBadCalibFilter);
