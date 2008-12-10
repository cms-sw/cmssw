#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerMaxSample_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerMaxSample_hh

#include "RecoLocalCalo/EcalRecProducers/interface/ESRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/ESRecHitSimAlgo.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace edm {
        class ParameterSet;
        class EventSetup;
        class Event;
}

class ESRecHitWorker : public ESRecHitWorkerBaseClass {

        public:
                ESRecHitWorker(const edm::ParameterSet& ps);
                ~ESRecHitWorker();

                void set(const edm::EventSetup& es);
                bool run(const edm::Event& evt, const ESDigiCollection::const_iterator & digi, ESRecHitCollection & result);

        private:

                ESRecHitSimAlgo *algo_;
};
#endif
