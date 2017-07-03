#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerAnalFit_HH
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitWorkerAnalFit_HH

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerRunOneDigiBase.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAnalFitAlgo.h"

#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"

namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
	class ParameterSetDescription;
}

class EcalUncalibRecHitWorkerAnalFit : public EcalUncalibRecHitWorkerRunOneDigiBase {

        public:
                EcalUncalibRecHitWorkerAnalFit(const edm::ParameterSet& ps,edm::ConsumesCollector& c);
                EcalUncalibRecHitWorkerAnalFit() {};
                ~EcalUncalibRecHitWorkerAnalFit() override {};

                void set(const edm::EventSetup& es) override;
                bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result) override;
		
		edm::ParameterSetDescription getAlgoDescription() override;

        private:
                EcalUncalibRecHitRecAnalFitAlgo<EBDataFrame> algoEB_;
                EcalUncalibRecHitRecAnalFitAlgo<EEDataFrame> algoEE_;

                
                edm::ESHandle<EcalGainRatios> pRatio;
                edm::ESHandle<EcalPedestals> pedHandle;
};
#endif
