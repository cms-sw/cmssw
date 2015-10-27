#include "RecoLocalCalo/HGCalRecProducers/plugins/HGCalUncalibRecHitWorkerWeights.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

HGCalUncalibRecHitWorkerWeights::HGCalUncalibRecHitWorkerWeights(const edm::ParameterSet&ps) :
  HGCalUncalibRecHitWorkerBaseClass(ps)
{
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
