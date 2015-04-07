#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalUncalibRecHitWorkerWeights.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

template< typename DET >
void configureIt(const edm::ParameterSet& conf, 
                 HGCalUncalibRecHitRecWeightsAlgo<HGCDataFrame<DET,HGCSample>>& maker) {
  constexpr char isSiFE[]           = "isSiFE";
  constexpr char mipInfC[]          = "mipInfC";
  constexpr char adcNbits[]         = "adcNbits";
  constexpr char adcSaturation_fC[] = "adcSaturation_fC";
  constexpr char tdcNbits[]         = "tdcNbits";
  constexpr char tdcSaturation_fC[] = "tdcSaturation_fC";
  constexpr char lsbInMIP[]         = "lsbInMIP";
  
  if( conf.exists(isSiFE) ) {
    maker.set_isSiFESim(conf.getParameter<bool>(isSiFE));
  } else {
    maker.set_isSiFESim(false);
  }

  if( conf.exists(adcNbits) ) {
    uint32_t nBits       = conf.getParameter<uint32_t>(adcNbits);
    double saturation_fC = conf.getParameter<double>(adcSaturation_fC);
    float adcLSB_fC      = saturation_fC/pow(2.,nBits);    
    maker.set_ADCLSBInfC(adcLSB_fC);
  } else {
    maker.set_ADCLSBInfC(-1.);
  }
  
  if( conf.exists(tdcNbits) ) {
    uint32_t nBits       = conf.getParameter<uint32_t>(tdcNbits);
    double saturation_fC = conf.getParameter<double>(tdcSaturation_fC);
    float tdcLSB_fC      = saturation_fC/pow(2.,nBits); 
    maker.set_TDCLSBInfC(tdcLSB_fC);
  } else {
    maker.set_TDCLSBInfC(-1.);
  } 
    
  if( conf.exists(mipInfC) ) {
    maker.set_fCToMIP(conf.getParameter<double>(mipInfC));
  } else {
    maker.set_fCToMIP(-1.);
  }
  
  if(conf.exists(lsbInMIP) ) {
    maker.set_ADCToMIP(conf.getParameter<double>(lsbInMIP));
  } else {
    maker.set_ADCToMIP(-1.);
  }
}

HGCalUncalibRecHitWorkerWeights::HGCalUncalibRecHitWorkerWeights(const edm::ParameterSet&ps) :
  HGCalUncalibRecHitWorkerBaseClass(ps)
{
  const edm::ParameterSet ee_cfg = ps.getParameterSet("HGCEEConfig");
  const edm::ParameterSet hef_cfg = ps.getParameterSet("HGCHEFConfig");
  const edm::ParameterSet heb_cfg = ps.getParameterSet("HGCHEBConfig");
  configureIt(ee_cfg,uncalibMaker_ee_);
  configureIt(hef_cfg,uncalibMaker_hef_);
  configureIt(heb_cfg,uncalibMaker_heb_);
}

void
HGCalUncalibRecHitWorkerWeights::set(const edm::EventSetup& es)
{
  
}


bool
HGCalUncalibRecHitWorkerWeights::run1( const edm::Event & evt,
                                       const HGCEEDigiCollection::const_iterator & itdg,
                                       HGCeeUncalibratedRecHitCollection & result )
{
  result.push_back(uncalibMaker_ee_.makeRecHit(*itdg));  
  return true;
}

bool
HGCalUncalibRecHitWorkerWeights::run2( const edm::Event & evt,
                const HGCHEDigiCollection::const_iterator & itdg,
                HGChefUncalibratedRecHitCollection & result )
{
  result.push_back(uncalibMaker_hef_.makeRecHit(*itdg));
  return true;
}

bool
HGCalUncalibRecHitWorkerWeights::run3( const edm::Event & evt,
                const HGCHEDigiCollection::const_iterator & itdg,
                HGChebUncalibratedRecHitCollection & result )
{
  result.push_back(uncalibMaker_heb_.makeRecHit(*itdg));
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( HGCalUncalibRecHitWorkerFactory, HGCalUncalibRecHitWorkerWeights, "HGCalUncalibRecHitWorkerWeights" );
