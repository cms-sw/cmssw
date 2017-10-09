#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerRunOneDigiBase_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerRunOneDigiBase_hh

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"


//! this is a compatibility base after the primary application (EcalUncalibRecHitProducer) moved to full collection used in argument
//! given reduced overhead by running on the full collection as input, 
//! derived classes with CPU constraints should move to EcalUncalibRecHitWorkerBaseClass

class EcalUncalibRecHitWorkerRunOneDigiBase : public EcalUncalibRecHitWorkerBaseClass {
 public:
  EcalUncalibRecHitWorkerRunOneDigiBase(const edm::ParameterSet& p, edm::ConsumesCollector& c) : EcalUncalibRecHitWorkerBaseClass(p,c){}
  EcalUncalibRecHitWorkerRunOneDigiBase(const edm::ParameterSet& p) : EcalUncalibRecHitWorkerBaseClass(p) {}
  EcalUncalibRecHitWorkerRunOneDigiBase() {}
  virtual ~EcalUncalibRecHitWorkerRunOneDigiBase() {}
  
  virtual bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result) = 0;

  virtual void run(const edm::Event& evt, const EcalDigiCollection & digis, EcalUncalibratedRecHitCollection & result) override
  {
    result.reserve(result.size() + digis.size());
    for (auto it = digis.begin(); it != digis.end(); ++it)
      run(evt, it, result);
  }

};

#endif
