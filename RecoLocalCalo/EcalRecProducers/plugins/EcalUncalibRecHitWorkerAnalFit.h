#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerAnalFit_HH
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerAnalFit_HH

#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Utilities/interface/ESGetToken.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerRunOneDigiBase.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAnalFitAlgo.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"

namespace edm {
  class Event;
  class EventSetup;
  class ParameterSet;
  class ParameterSetDescription;
}  // namespace edm

class EcalUncalibRecHitWorkerAnalFit : public EcalUncalibRecHitWorkerRunOneDigiBase {
public:
  EcalUncalibRecHitWorkerAnalFit(const edm::ParameterSet& ps, edm::ConsumesCollector& c);
  EcalUncalibRecHitWorkerAnalFit(){};
  ~EcalUncalibRecHitWorkerAnalFit() override{};

  void set(const edm::EventSetup& es) override;
  bool run(const edm::Event& evt,
           const EcalDigiCollection::const_iterator& digi,
           EcalUncalibratedRecHitCollection& result) override;

  edm::ParameterSetDescription getAlgoDescription() override;

private:
  EcalUncalibRecHitRecAnalFitAlgo<EBDataFrame> algoEB_;
  EcalUncalibRecHitRecAnalFitAlgo<EEDataFrame> algoEE_;

  edm::ESHandle<EcalGainRatios> pRatio;
  edm::ESHandle<EcalPedestals> pedHandle;
  edm::ESGetToken<EcalGainRatios, EcalGainRatiosRcd> ratiosToken_;
  edm::ESGetToken<EcalPedestals, EcalPedestalsRcd> pedestalsToken_;
};
#endif
