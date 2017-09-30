// -*- C++ -*-
//
// Package:    ecalBadCalibSep2017ListFilter
// Class:      ecalBadCalibSep2017ListFilter
//
/**\class ecalBadCalibSep2017ListFilter ecalBadCalibSep2017ListFilter.cc
 
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
 
using namespace std;
 
 
class ecalBadCalibSep2017ListFilter : public edm::global::EDFilter<> {
 
public:
 
  explicit ecalBadCalibSep2017ListFilter(const edm::ParameterSet & iConfig);
  ~ecalBadCalibSep2017ListFilter() override {}
 
private:
 
  // main filter function
 
 bool filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const override; 
 
  // input parameters
 
  // eb rechit collection (from AOD)
  const edm::EDGetTokenT<EcalRecHitCollection>  ebRHSrcToken_;  
  // ee rechit collection (from AOD)
  const edm::EDGetTokenT<EcalRecHitCollection>  eeRHSrcToken_;

  //config parameters (defining the cuts on the bad SCs)
  const double ebMin_;              // EB rechit et threshold
  const double eeMin_;              // EE rechit et threshold
 
  const std::vector<unsigned int> baddetEB_;    // DetIds of bad EB channels  
  const std::vector<unsigned int> baddetEE_;    // DetIds of bad EE channels
 
  const bool taggingMode_;
  const bool debug_;                // prints out debug info if set to true
 
};
 
// read the parameters from the config file
ecalBadCalibSep2017ListFilter::ecalBadCalibSep2017ListFilter(const edm::ParameterSet & iConfig)
  : ebRHSrcToken_     (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EBRecHitSource")))
  , eeRHSrcToken_     (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitSource")))
  , ebMin_          (iConfig.getParameter<double>("ebMinEt"))
  , eeMin_          (iConfig.getParameter<double>("eeMinEt"))
  , baddetEB_       (iConfig.getParameter<std::vector<unsigned int> >("baddetEB"))
  , baddetEE_       (iConfig.getParameter<std::vector<unsigned int> >("baddetEE"))
  , taggingMode_    (iConfig.getParameter<bool>("taggingMode"))
  , debug_          (iConfig.getParameter<bool>("debug"))
{
  produces<bool>();
}
 

bool ecalBadCalibSep2017ListFilter::filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {


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
 
  for (const auto eeit : baddetEE_) {
    
    EEDetId eedet(eeit);
    
    if (eedet.rawId()==0) continue;
     
    // find rechit corresponding to this DetId
    EcalRecHitCollection::const_iterator eehit=eeRHs->find(eedet);
 
    if (eehit==eeRHs->end()) continue;
 
    // if rechit not found, move to next DetId   
    if (eehit->id().rawId()==0 || eehit->id().rawId()!= eedet.rawId()) { continue; }
     
    
    // rechit has been found: obtain crystal coordinates, energy 
    ix=eedet.ix();
    iy=eedet.iy();
    iz=eedet.zside();
    ene=eehit->energy();
 
    // compute transverse energy
    GlobalPoint posee=pG->getPosition(eedet);
    float pf = posee.perp()/posee.mag();
    et=ene*pf;
    
    // print some debug info
    if (debug_) {
      edm::LogInfo("ecalBadCalibSep2017ListFilter") << "DetId=" <<  eedet.rawId();
      edm::LogInfo("ecalBadCalibSep2017ListFilter") << "ix=" << ix << " iy=" << iy << " iz=" << iz;
      edm::LogInfo("ecalBadCalibSep2017ListFilter") << "Et=" << et << " thresh=" << eeMin_;
    }
       
       
    // if transverse energy is above threshold and channel has bad IC 
    if (et>eeMin_) {
      pass=false;
      if (debug_) {
	edm::LogInfo("ecalBadCalibSep2017ListFilter") << "DUMP EVENT" << std::endl;
      }
    }

  }
  
 
  // print the decision if event is bad
  if (pass==false && debug_) edm::LogInfo("ecalBadCalibSep2017ListFilter") << "REJECT EVENT!!!";
   
  iEvent.put(std::make_unique<bool>(pass));
 
  return taggingMode_ || pass;
}
 
 
#include "FWCore/Framework/interface/MakerMacros.h"
 
DEFINE_FWK_MODULE(ecalBadCalibSep2017ListFilter);
