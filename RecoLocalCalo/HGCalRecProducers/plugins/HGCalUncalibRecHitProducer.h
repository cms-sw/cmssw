#ifndef RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitProducer_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalUncalibRecHitProducer_hh

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "DataFormats/HGCDigi/interface/HGCDataFrame.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalUncalibRecHitWorkerBaseClass.h"


class HGCalUncalibRecHitProducer : public edm::EDProducer {

        public:
                explicit HGCalUncalibRecHitProducer(const edm::ParameterSet& ps);
                ~HGCalUncalibRecHitProducer();
                virtual void produce(edm::Event& evt, const edm::EventSetup& es);

        private:

                edm::InputTag eeDigiCollection_; // collection of HGCEE digis
                edm::InputTag hefDigiCollection_; // collection of HGCHEF digis
                edm::InputTag hebDigiCollection_; // collection of HGCHEB digis

                std::string eeHitCollection_; // secondary name to be given to HGCEE collection of hits
                std::string hefHitCollection_; // secondary name to be given to HGCHEF collection of hits
                std::string hebHitCollection_; // secondary name to be given to HGCHEB collection of hits

                HGCalUncalibRecHitWorkerBaseClass * worker_;
};
#endif
