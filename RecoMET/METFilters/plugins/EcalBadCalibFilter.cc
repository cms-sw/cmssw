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
 
using namespace std;
 
 
class EcalBadCalibFilter : public edm::global::EDFilter<> {
 
public:
 
  explicit EcalBadCalibFilter(const edm::ParameterSet & iConfig);
  ~EcalBadCalibFilter() override {}
 
private:
 
  // main filter function
 
 bool filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const override; 
 
  // input parameters
  // ecal rechit collection (from AOD)
  const edm::EDGetTokenT<EcalRecHitCollection>  eeRHSrcToken_;

  //config parameters (defining the cuts on the bad SCs)
  const double eeMin_;              // ecal rechit et threshold
 
  const std::vector<unsigned int> baddetEE_;    // DetIds of bad Ecal channels
 
  const bool taggingMode_;
  const bool debug_;                // prints out debug info if set to true
 
};
 
// read the parameters from the config file
EcalBadCalibFilter::EcalBadCalibFilter(const edm::ParameterSet & iConfig) 
  : eeRHSrcToken_   (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EERecHitSource")))
  , eeMin_          (iConfig.getParameter<double>("eeMinEt"))
  , baddetEE_       (iConfig.getParameter<std::vector<unsigned int> >("baddetEE"))
  , taggingMode_    (iConfig.getParameter<bool>("taggingMode"))
  , debug_          (iConfig.getParameter<bool>("debug"))
{
  produces<bool>();
}
 

bool EcalBadCalibFilter::filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {


  // load required collections
 
   // Ecal rechit collection
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
      edm::LogInfo("EcalBadCalibFilter") << "DetId=" <<  eedet.rawId();
      edm::LogInfo("EcalBadCalibFilter") << "ix=" << ix << " iy=" << iy << " iz=" << iz;
      edm::LogInfo("EcalBadCalibFilter") << "Et=" << et << " thresh=" << eeMin_;
    }
       
       
    // if transverse energy is above threshold and channel has bad IC 
    if (et>eeMin_) {
      pass=false;
      if (debug_) {
	edm::LogInfo("EcalBadCalibFilter") << "DUMP EVENT" << std::endl;
      }
    }

  }
  
 
  // print the decision if event is bad
  if (pass==false && debug_) edm::LogInfo("EcalBadCalibFilter") << "REJECT EVENT!!!";
   
  iEvent.put(std::make_unique<bool>(pass));
 
  return taggingMode_ || pass;
}
 
 
#include "FWCore/Framework/interface/MakerMacros.h"
 
DEFINE_FWK_MODULE(EcalBadCalibFilter);
