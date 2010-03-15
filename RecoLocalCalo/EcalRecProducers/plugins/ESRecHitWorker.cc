#include "RecoLocalCalo/EcalRecProducers/plugins/ESRecHitWorker.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "CondFormats/DataRecord/interface/ESGainRcd.h"
#include "CondFormats/DataRecord/interface/ESChannelStatusRcd.h"
#include "CondFormats/DataRecord/interface/ESMIPToGeVConstantRcd.h"
#include "CondFormats/DataRecord/interface/ESTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/ESPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/ESIntercalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/ESRecHitRatioCutsRcd.h"

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

ESRecHitWorker::ESRecHitWorker(const edm::ParameterSet& ps) :
        ESRecHitWorkerBaseClass( ps )
{
  //These should be taken from a DB
  recoAlgo_ = ps.getParameter<int>("ESRecoAlgo");

  if (recoAlgo_ == 0)
    algoW_ = new ESRecHitSimAlgo();
  else 
    algoF_ = new ESRecHitFitAlgo();
}

ESRecHitWorker::~ESRecHitWorker() {
  if (recoAlgo_ == 0)
    delete algoW_;
  else
    delete algoF_;
}

void ESRecHitWorker::set(const edm::EventSetup& es) {

  es.get<ESGainRcd>().get(esgain_);
  const ESGain *gain = esgain_.product();

  es.get<ESMIPToGeVConstantRcd>().get(esMIPToGeV_);
  const ESMIPToGeVConstant *mipToGeV = esMIPToGeV_.product();

  double ESGain = gain->getESGain();
  double ESMIPToGeV = (ESGain == 1) ? mipToGeV->getESValueLow() : mipToGeV->getESValueHigh(); 

  es.get<ESTBWeightsRcd>().get(esWeights_);
  const ESTBWeights *wgts = esWeights_.product();

  ESTBWeights::ESTBWeightMap::const_iterator wit = wgts->getMap().find( ESGain );
  ESWeightSet wset;
  double w0 = 0;
  double w1 = 0;
  double w2 = 0;
  if ( wit != wgts->getMap().end() ) {
    wset = wit->second;
    const ESWeightSet::ESWeightMatrix& mat = wset.getWeights();
    w0 = mat(int(ESGain-1), 0);
    w1 = mat(int(ESGain-1), 1);
    w2 = mat(int(ESGain-1), 2);
  }

  es.get<ESPedestalsRcd>().get(esPedestals_);
  const ESPedestals *peds = esPedestals_.product();

  es.get<ESIntercalibConstantsRcd>().get(esMIPs_);
  const ESIntercalibConstants *mips = esMIPs_.product();

  es.get<ESChannelStatusRcd>().get(esChannelStatus_);
  const ESChannelStatus *channelStatus = esChannelStatus_.product();

  es.get<ESRecHitRatioCutsRcd>().get(esRatioCuts_);
  const ESRecHitRatioCuts *ratioCuts = esRatioCuts_.product(); 

  if (recoAlgo_ == 0) {
    algoW_->setESGain(ESGain);
    algoW_->setMIPGeV(ESMIPToGeV);
    algoW_->setW0(w0);
    algoW_->setW1(w1);
    algoW_->setW2(w2);
    algoW_->setPedestals(*peds);
    algoW_->setIntercalibConstants(*mips);
    algoW_->setChannelStatus(*channelStatus);
    algoW_->setRatioCuts(*ratioCuts);
  } else {
    algoF_->setESGain(ESGain);
    algoF_->setMIPGeV(ESMIPToGeV);
    algoF_->setPedestals(*peds);
    algoF_->setIntercalibConstants(*mips);
    algoF_->setChannelStatus(*channelStatus);
    algoF_->setRatioCuts(*ratioCuts);
  }
}

bool
ESRecHitWorker::run( const edm::Event & evt, 
                const ESDigiCollection::const_iterator & itdg, 
                ESRecHitCollection & result )
{
  if (recoAlgo_ == 0)
    result.push_back( algoW_->reconstruct(*itdg) );
  else 
    result.push_back( algoF_->reconstruct(*itdg) );
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( ESRecHitWorkerFactory, ESRecHitWorker, "ESRecHitWorker" );
