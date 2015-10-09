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
  HGCEEmipInKeV_ =  ps.getParameter<double>("HGCEEmipInKeV");
  HGCEElsbInMIP_ =  ps.getParameter<double>("HGCEElsbInMIP");
  HGCEEmip2noise_ = ps.getParameter<double>("HGCEEmip2noise");
  hgceeADCtoGeV_ = HGCEEmipInKeV_ * HGCEElsbInMIP_*keV2GeV; 
  // HGChef constants
  HGCHEFmipInKeV_ =  ps.getParameter<double>("HGCHEFmipInKeV");
  HGCHEFlsbInMIP_ =  ps.getParameter<double>("HGCHEFlsbInMIP");
  HGCHEFmip2noise_ = ps.getParameter<double>("HGCHEFmip2noise");
  hgchefADCtoGeV_ = HGCHEFmipInKeV_ * HGCHEFlsbInMIP_*keV2GeV;
  // HGCheb constants
  HGCHEBmipInKeV_ =  ps.getParameter<double>("HGCHEBmipInKeV");
  HGCHEBlsbInMIP_ =  ps.getParameter<double>("HGCHEBlsbInMIP");
  HGCHEBmip2noise_ = ps.getParameter<double>("HGCHEBmip2noise");
  hgchebADCtoGeV_ = HGCHEBmipInKeV_ * HGCHEBlsbInMIP_*keV2GeV;
}

void HGCalRecHitWorkerSimple::set(const edm::EventSetup& es) {
}


bool
HGCalRecHitWorkerSimple::run( const edm::Event & evt,
                              const HGCUncalibratedRecHit& uncalibRH,
                              HGCRecHitCollection & result )
{
  DetId detid=uncalibRH.id();  
  uint32_t recoFlag = 0;
    
  switch( detid.subdetId() ) {
  case HGCEE:
    rechitMaker_->setADCToGeVConstant(float(hgceeADCtoGeV_) );
    break;
  case HGCHEF:
    rechitMaker_->setADCToGeVConstant(float(hgchefADCtoGeV_) );
    break;
  case HGCHEB:
    rechitMaker_->setADCToGeVConstant(float(hgchebADCtoGeV_) );
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
