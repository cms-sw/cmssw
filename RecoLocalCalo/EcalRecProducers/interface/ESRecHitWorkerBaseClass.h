#ifndef RecoLocalCalo_EcalRecAlgos_ESRecHitWorkerBaseClass_hh
#define RecoLocalCalo_EcalRecAlgos_ESRecHitWorkerBaseClass_hh

#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "FWCore/Framework/interface/EventSetup.h"

namespace edm {
        class ParameterSet;
}

class ESRecHitWorkerBaseClass {
        public:
                ESRecHitWorkerBaseClass(const edm::ParameterSet&) {};
                virtual ~ESRecHitWorkerBaseClass() {};

                virtual void set(const edm::EventSetup& es) = 0;
                virtual bool run(const ESDigiCollection::const_iterator& digi, ESRecHitCollection & result) = 0;
};

#endif
