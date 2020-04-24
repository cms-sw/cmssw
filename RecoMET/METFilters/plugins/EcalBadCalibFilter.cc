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
#include "DataFormats/EcalDetId/interface/EBDetId.h"
#include "DataFormats/EcalDetId/interface/EcalSubdetector.h"

 
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
  const edm::EDGetTokenT<EcalRecHitCollection>  ecalRHSrcToken_;

  //config parameters (defining the cuts on the bad SCs)
  const double ecalMin_;              // ecal rechit et threshold
 
  const std::vector<unsigned int> baddetEcal_;    // DetIds of bad Ecal channels
 
  const bool taggingMode_;
  const bool debug_;                // prints out debug info if set to true
 
};
 
// read the parameters from the config file
EcalBadCalibFilter::EcalBadCalibFilter(const edm::ParameterSet & iConfig) 
  : ecalRHSrcToken_   (consumes<EcalRecHitCollection>(iConfig.getParameter<edm::InputTag>("EcalRecHitSource")))
  , ecalMin_          (iConfig.getParameter<double>("ecalMinEt"))
  , baddetEcal_       (iConfig.getParameter<std::vector<unsigned int> >("baddetEcal"))
  , taggingMode_    (iConfig.getParameter<bool>("taggingMode"))
  , debug_          (iConfig.getParameter<bool>("debug"))
{
  produces<bool>();
}
 

bool EcalBadCalibFilter::filter(edm::StreamID, edm::Event & iEvent, const edm::EventSetup & iSetup) const {


  // load required collections
 
   // Ecal rechit collection
  edm::Handle<EcalRecHitCollection> ecalRHs;
  iEvent.getByToken(ecalRHSrcToken_, ecalRHs);
 
  // Calo Geometry - needed for computing E_t
  edm::ESHandle<CaloGeometry> pG;
  iSetup.get<CaloGeometryRecord>().get(pG);
 
  
  // by default the event is OK
  bool pass = true;

  for (const auto ecalit : baddetEcal_) {

    DetId ecaldet(ecalit);
    
    if (ecaldet.rawId()==0) continue;
     
    // find rechit corresponding to this DetId
    EcalRecHitCollection::const_iterator ecalhit=ecalRHs->find(ecaldet);
 
    if (ecalhit==ecalRHs->end()) continue;
 
    // if rechit not found, move to next DetId   
    if (ecalhit->id().rawId()==0 || ecalhit->id().rawId()!= ecaldet.rawId()) { continue; }
    

    // define energy variables
    float ene=0;
    float et=0;

    // rechit has been found: obtain crystal energy 
    ene=ecalhit->energy();
 
    // compute transverse energy
    GlobalPoint posecal=pG->getPosition(ecaldet);
    float pf = posecal.perp()/posecal.mag();
    et=ene*pf;
    

    // print some debug info
    if (debug_) {
      
      int ix,iy,iz;
      ix=0,iy=0,iz=0;
      
      // ref: DataFormats/EcalDetId/interface/EcalSubdetector.h
      // EcalBarrel
      if (ecaldet.subdetId()==1) {
	EBDetId ebdet(ecalit);
	ix=ebdet.ieta();
	iy=ebdet.iphi();
	iz=ebdet.zside();
	
	edm::LogInfo("EcalBadCalibFilter") << "DetId=" <<  ecaldet.rawId();
	edm::LogInfo("EcalBadCalibFilter") << "ieta=" << ix << " iphi=" << iy << " iz=" << iz;
	edm::LogInfo("EcalBadCalibFilter") << "Et=" << et << " thresh=" << ecalMin_;
      }

      // EcalEndcap
      if (ecaldet.subdetId()==2) {
	EEDetId eedet(ecalit);
	ix=eedet.ix();
	iy=eedet.iy();
	iz=eedet.zside();

	edm::LogInfo("EcalBadCalibFilter") << "DetId=" <<  ecaldet.rawId();
	edm::LogInfo("EcalBadCalibFilter") << "ix=" << ix << " iy=" << iy << " iz=" << iz;
	edm::LogInfo("EcalBadCalibFilter") << "Et=" << et << " thresh=" << ecalMin_;
      }
     
    }
       
       
    // if transverse energy is above threshold and channel has bad IC 
    if (et>ecalMin_) {
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
