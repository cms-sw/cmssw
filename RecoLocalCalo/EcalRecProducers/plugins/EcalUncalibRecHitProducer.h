#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitProducer_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitProducer_hh

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/EcalDigi/interface/EEDataFrame.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"


class EcalUncalibRecHitProducer : public edm::EDProducer {

        public:
                explicit EcalUncalibRecHitProducer(const edm::ParameterSet& ps);
                ~EcalUncalibRecHitProducer();
                virtual void produce(edm::Event& evt, const edm::EventSetup& es);

        private:

                edm::InputTag ebDigiCollection_; // collection of EB digis
                edm::InputTag eeDigiCollection_; // collection of EE digis

                std::string ebHitCollection_; // secondary name to be given to collection of hits
                std::string eeHitCollection_; // secondary name to be given to collection of hits

                EcalUncalibRecHitWorkerBaseClass * worker_;
};
#endif
