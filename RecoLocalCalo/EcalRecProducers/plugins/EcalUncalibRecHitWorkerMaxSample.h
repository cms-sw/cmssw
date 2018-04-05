#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerMaxSample_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerMaxSample_hh

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerRunOneDigiBase.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMaxSampleAlgo.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace edm {
        class ParameterSet;
        class EventSetup;
        class Event;
	class ParameterSetDescription;
}

class EcalUncalibRecHitWorkerMaxSample : public EcalUncalibRecHitWorkerRunOneDigiBase {

 public:
  EcalUncalibRecHitWorkerMaxSample(const edm::ParameterSet& ps,edm::ConsumesCollector& c);
  EcalUncalibRecHitWorkerMaxSample() {};
  ~EcalUncalibRecHitWorkerMaxSample() override {};
  
  void set(const edm::EventSetup& es) override;
  bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result) override;
  
  edm::ParameterSetDescription getAlgoDescription() override;
 private:
  
  EcalUncalibRecHitMaxSampleAlgo<EBDataFrame> ebAlgo_;
  EcalUncalibRecHitMaxSampleAlgo<EEDataFrame> eeAlgo_;
};
#endif
