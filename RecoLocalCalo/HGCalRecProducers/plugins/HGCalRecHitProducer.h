#ifndef RecoLocalCalo_HGCalRecProducers_HGCalRecHitProducer_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalRecHitProducer_hh

/** \class HGCalRecHitProducer
 *   produce HGCAL rechits from uncalibrated rechits
 *
 *  simplified version of Ecal code
 *
 *  \author Valeri Andreev
 *
 **/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerBaseClass.h"

class HGCalRecHitProducer : public edm::EDProducer {

        public:
                explicit HGCalRecHitProducer(const edm::ParameterSet& ps);
                ~HGCalRecHitProducer();
                virtual void produce(edm::Event& evt, const edm::EventSetup& es);

        private:

                edm::InputTag eeUncalibRecHitCollection_; // secondary name given to collection of HGCEE uncalib rechits
                edm::InputTag hefUncalibRecHitCollection_; // secondary name given to collection of HGCHEF uncalib rechits
                edm::InputTag hebUncalibRecHitCollection_; // secondary name given to collection of HGCHEB uncalib rechits
                std::string eeRechitCollection_; // secondary name to be given to HGCEE collection of hits
                std::string hefRechitCollection_; // secondary name to be given to HGCHEF collection of hits
                std::string hebRechitCollection_; // secondary name to be given to HGCHEB collection of hits

                bool killDeadChannels_;

                HGCalRecHitWorkerBaseClass * worker_;

};

#endif
