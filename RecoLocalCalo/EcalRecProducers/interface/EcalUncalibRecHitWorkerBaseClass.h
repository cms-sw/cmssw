#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerBaseClass_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerBaseClass_hh

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/ConsumesCollector.h"
//#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
	class ConfigurationDescriptions;
}

class EcalUncalibRecHitWorkerBaseClass {
 public:
  EcalUncalibRecHitWorkerBaseClass(const edm::ParameterSet&, edm::ConsumesCollector& c) {}
  EcalUncalibRecHitWorkerBaseClass(const edm::ParameterSet&) {}
  EcalUncalibRecHitWorkerBaseClass() {}
  virtual ~EcalUncalibRecHitWorkerBaseClass() {}
  
  virtual void set(const edm::EventSetup& es) = 0;//{};
  virtual void set(const edm::Event& evt) {}
  virtual bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result) = 0; //{ return false; };
  virtual void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {};
};

#endif
