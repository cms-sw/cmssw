#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitWorkerSimple.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"

HGCalRecHitWorkerSimple::HGCalRecHitWorkerSimple(const edm::ParameterSet&ps) :
  HGCalRecHitWorkerBaseClass(ps) {
  rechitMaker_.reset( new HGCalRecHitSimpleAlgo() );
  constexpr float keV2GeV = 1e-6;
  // HGCee constants 
  HGCEE_keV2DIGI_   =  ps.getParameter<double>("HGCEE_keV2DIGI");
  hgceeUncalib2GeV_ = keV2GeV/HGCEE_keV2DIGI_;
  
  // HGChef constants
  HGCHEF_keV2DIGI_   =  ps.getParameter<double>("HGCHEF_keV2DIGI");
  hgchefUncalib2GeV_ = keV2GeV/HGCHEF_keV2DIGI_;
  
  // HGCheb constants
  HGCHEB_keV2DIGI_   =  ps.getParameter<double>("HGCHEB_keV2DIGI");
  hgchebUncalib2GeV_ = keV2GeV/HGCHEB_keV2DIGI_;
}

void HGCalRecHitWorkerSimple::set(const edm::EventSetup& es) {
}


bool
HGCalRecHitWorkerSimple::run( const edm::Event & evt,
                              const HGCUncalibratedRecHit& uncalibRH,
                              HGCRecHitCollection & result ) {
  DetId detid=uncalibRH.id();  
  uint32_t recoFlag = 0;
    
  switch( detid.subdetId() ) {
  case HGCEE:
    rechitMaker_->setADCToGeVConstant(float(hgceeUncalib2GeV_) );
    break;
  case HGCHEF:
    rechitMaker_->setADCToGeVConstant(float(hgchefUncalib2GeV_) );
    break;
  case HGCHEB:
    rechitMaker_->setADCToGeVConstant(float(hgchebUncalib2GeV_) );
    break;
  default:
    throw cms::Exception("NonHGCRecHit")
      << "Rechit with detid = " << detid.rawId() << " is not HGC!";
  }
  
  // make the rechit and put in the output collection
  if (recoFlag == 0) {
    HGCRecHit myrechit( rechitMaker_->makeRecHit(uncalibRH, 0) );    
    result.push_back(myrechit);
  }

  return true;
}

HGCalRecHitWorkerSimple::~HGCalRecHitWorkerSimple(){
}


#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( HGCalRecHitWorkerFactory, HGCalRecHitWorkerSimple, "HGCalRecHitWorkerSimple" );
