#include "ESRecHitWorker.h"

#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalUncalibratedRecHit.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <cmath>
#include <iomanip>
#include <iostream>

ESRecHitWorker::ESRecHitWorker(const edm::ParameterSet &ps, edm::ConsumesCollector cc) : ESRecHitWorkerBaseClass(ps) {
  recoAlgo_ = ps.getParameter<int>("ESRecoAlgo");
  esgainToken_ = cc.esConsumes<ESGain, ESGainRcd>();
  esMIPToGeVToken_ = cc.esConsumes<ESMIPToGeVConstant, ESMIPToGeVConstantRcd>();
  esWeightsToken_ = cc.esConsumes<ESTimeSampleWeights, ESTimeSampleWeightsRcd>();
  esPedestalsToken_ = cc.esConsumes<ESPedestals, ESPedestalsRcd>();
  esMIPsToken_ = cc.esConsumes<ESIntercalibConstants, ESIntercalibConstantsRcd>();
  esChannelStatusToken_ = cc.esConsumes<ESChannelStatus, ESChannelStatusRcd>();
  esRatioCutsToken_ = cc.esConsumes<ESRecHitRatioCuts, ESRecHitRatioCutsRcd>();
  esAngleCorrFactorsToken_ = cc.esConsumes<ESAngleCorrectionFactors, ESAngleCorrectionFactorsRcd>();

  if (recoAlgo_ == 0)
    algoW_ = new ESRecHitSimAlgo();
  else if (recoAlgo_ == 1)
    algoF_ = new ESRecHitFitAlgo();
  else
    algoA_ = new ESRecHitAnalyticAlgo();
}

ESRecHitWorker::~ESRecHitWorker() {
  if (recoAlgo_ == 0)
    delete algoW_;
  else if (recoAlgo_ == 1)
    delete algoF_;
  else
    delete algoA_;
}

void ESRecHitWorker::set(const edm::EventSetup &es) {
  esgain_ = es.getHandle(esgainToken_);
  const ESGain *gain = esgain_.product();

  esMIPToGeV_ = es.getHandle(esMIPToGeVToken_);
  const ESMIPToGeVConstant *mipToGeV = esMIPToGeV_.product();

  double ESGain = gain->getESGain();
  double ESMIPToGeV = (ESGain == 1) ? mipToGeV->getESValueLow() : mipToGeV->getESValueHigh();

  esWeights_ = es.getHandle(esWeightsToken_);
  const ESTimeSampleWeights *wgts = esWeights_.product();

  float w0 = wgts->getWeightForTS0();
  float w1 = wgts->getWeightForTS1();
  float w2 = wgts->getWeightForTS2();

  esPedestals_ = es.getHandle(esPedestalsToken_);
  const ESPedestals *peds = esPedestals_.product();

  esMIPs_ = es.getHandle(esMIPsToken_);
  const ESIntercalibConstants *mips = esMIPs_.product();

  esAngleCorrFactors_ = es.getHandle(esAngleCorrFactorsToken_);
  const ESAngleCorrectionFactors *ang = esAngleCorrFactors_.product();

  esChannelStatus_ = es.getHandle(esChannelStatusToken_);
  const ESChannelStatus *channelStatus = esChannelStatus_.product();

  esRatioCuts_ = es.getHandle(esRatioCutsToken_);
  const ESRecHitRatioCuts *ratioCuts = esRatioCuts_.product();

  if (recoAlgo_ == 0) {
    algoW_->setESGain(ESGain);
    algoW_->setMIPGeV(ESMIPToGeV);
    algoW_->setW0(w0);
    algoW_->setW1(w1);
    algoW_->setW2(w2);
    algoW_->setPedestals(peds);
    algoW_->setIntercalibConstants(mips);
    algoW_->setChannelStatus(channelStatus);
    algoW_->setRatioCuts(ratioCuts);
    algoW_->setAngleCorrectionFactors(ang);
  } else if (recoAlgo_ == 1) {
    algoF_->setESGain(ESGain);
    algoF_->setMIPGeV(ESMIPToGeV);
    algoF_->setPedestals(peds);
    algoF_->setIntercalibConstants(mips);
    algoF_->setChannelStatus(channelStatus);
    algoF_->setRatioCuts(ratioCuts);
    algoF_->setAngleCorrectionFactors(ang);
  } else {
    algoA_->setESGain(ESGain);
    algoA_->setMIPGeV(ESMIPToGeV);
    algoA_->setPedestals(peds);
    algoA_->setIntercalibConstants(mips);
    algoA_->setChannelStatus(channelStatus);
    algoA_->setRatioCuts(ratioCuts);
    algoA_->setAngleCorrectionFactors(ang);
  }
}

bool ESRecHitWorker::run(const ESDigiCollection::const_iterator &itdg, ESRecHitCollection &result) {
  if (recoAlgo_ == 0)
    result.push_back(algoW_->reconstruct(*itdg));
  else if (recoAlgo_ == 1)
    result.push_back(algoF_->reconstruct(*itdg));
  else
    result.push_back(algoA_->reconstruct(*itdg));
  return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN(ESRecHitWorkerFactory, ESRecHitWorker, "ESRecHitWorker");
