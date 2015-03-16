#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerMaxSample_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerMaxSample_hh

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMaxSampleAlgo.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace edm {
        class ParameterSet;
        class EventSetup;
        class Event;
	class ParameterSetDescription;
}

class EcalUncalibRecHitWorkerMaxSample : public EcalUncalibRecHitWorkerBaseClass {

 public:
  EcalUncalibRecHitWorkerMaxSample(const edm::ParameterSet& ps,edm::ConsumesCollector& c);
  EcalUncalibRecHitWorkerMaxSample() {};
  virtual ~EcalUncalibRecHitWorkerMaxSample() {};
  
  void set(const edm::EventSetup& es);
  bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result);
  
  edm::ParameterSetDescription getAlgoDescription();
 private:
  
  EcalUncalibRecHitMaxSampleAlgo<EBDataFrame> ebAlgo_;
  EcalUncalibRecHitMaxSampleAlgo<EEDataFrame> eeAlgo_;
};
#endif
