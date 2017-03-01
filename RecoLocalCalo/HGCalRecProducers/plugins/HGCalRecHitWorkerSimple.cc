#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalRecHitWorkerSimple.h"
#include "DataFormats/ForwardDetId/interface/HGCEEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCHEDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalDetId.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "CommonTools/Utils/interface/StringToEnumValue.h"
#include "DataFormats/HcalDetId/interface/HcalSubdetector.h"

HGCalRecHitWorkerSimple::HGCalRecHitWorkerSimple(const edm::ParameterSet&ps) :
  HGCalRecHitWorkerBaseClass(ps) {
  rechitMaker_.reset( new HGCalRecHitSimpleAlgo() );
  constexpr float keV2GeV = 1e-6;
  // HGCee constants 
  HGCEE_keV2DIGI_   =  ps.getParameter<double>("HGCEE_keV2DIGI");
  HGCEE_fCPerMIP_   =  ps.getParameter<std::vector<double> >("HGCEE_fCPerMIP");
  HGCEE_isSiFE_     =  ps.getParameter<bool>("HGCEE_isSiFE");
  hgceeUncalib2GeV_ = keV2GeV/HGCEE_keV2DIGI_;
  
  // HGChef constants
  HGCHEF_keV2DIGI_   =  ps.getParameter<double>("HGCHEF_keV2DIGI");
  HGCHEF_fCPerMIP_   =  ps.getParameter<std::vector<double> >("HGCHEF_fCPerMIP");
  HGCHEF_isSiFE_     =  ps.getParameter<bool>("HGCHEF_isSiFE");
  hgchefUncalib2GeV_ = keV2GeV/HGCHEF_keV2DIGI_;
  
  // HGCheb constants
  HGCHEB_keV2DIGI_   =  ps.getParameter<double>("HGCHEB_keV2DIGI");
  HGCHEB_isSiFE_     =  ps.getParameter<bool>("HGCHEB_isSiFE");
  hgchebUncalib2GeV_ = keV2GeV/HGCHEB_keV2DIGI_;

  // layer weights (from Valeri/Arabella)
  std::vector<float> weights;
  const auto& dweights = ps.getParameter<std::vector<double> >("layerWeights");
  for( auto weight : dweights ) {
    weights.push_back(weight);
  }
  rechitMaker_->setLayerWeights(weights);

  // residual correction for cell thickness
  const auto& rcorr = ps.getParameter<std::vector<double> >("thicknessCorrection");
  rcorr_.clear();
  rcorr_.push_back(1.f);
  for( auto corr : rcorr ) {
    rcorr_.push_back(1.0/corr);
  }
  
}

void HGCalRecHitWorkerSimple::set(const edm::EventSetup& es) {
  if (HGCEE_isSiFE_) {
    edm::ESHandle<HGCalGeometry> hgceeGeoHandle; 
    es.get<IdealGeometryRecord>().get("HGCalEESensitive",hgceeGeoHandle); 
    ddds_[0] = &(hgceeGeoHandle->topology().dddConstants());
  } else {
    ddds_[0] = nullptr;
  }
  if (HGCHEF_isSiFE_) {
    edm::ESHandle<HGCalGeometry> hgchefGeoHandle; 
    es.get<IdealGeometryRecord>().get("HGCalHESiliconSensitive",hgchefGeoHandle); 
    ddds_[1] = &(hgchefGeoHandle->topology().dddConstants());
  } else {
    ddds_[1] = nullptr;
  }
  ddds_[2] = nullptr;  
}


bool
HGCalRecHitWorkerSimple::run( const edm::Event & evt,
                              const HGCUncalibratedRecHit& uncalibRH,
                              HGCRecHitCollection & result ) {
  DetId detid=uncalibRH.id();  
  uint32_t recoFlag = 0;
  //const std::vector<double>* fCPerMIP = nullptr;
    
  switch( detid.subdetId() ) {
  case HGCEE:
    rechitMaker_->setADCToGeVConstant(float(hgceeUncalib2GeV_) );
    //fCPerMIP = &HGCEE_fCPerMIP_;
    break;
  case HGCHEF:
    rechitMaker_->setADCToGeVConstant(float(hgchefUncalib2GeV_) );
    //fCPerMIP = &HGCHEF_fCPerMIP_;
    break;
  case HcalEndcap:
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
    int thk = -1;
    if( detid.subdetId() != HcalEndcap ) {
      HGCalDetId hid(detid);
      thk = ddds_[hid.subdetId()-3]->waferTypeL(hid.wafer());
      // units out of rechit maker are MIP * (GeV/fC)
      // so multiple
    }
    const double new_E = myrechit.energy()*(thk == -1 ? 1.0 : rcorr_[thk]);
    myrechit.setEnergy(new_E);
    result.push_back(myrechit);
  }

  return true;
}

HGCalRecHitWorkerSimple::~HGCalRecHitWorkerSimple(){
}


#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( HGCalRecHitWorkerFactory, HGCalRecHitWorkerSimple, "HGCalRecHitWorkerSimple" );
