#ifndef RecoLocalCalo_EcalRecProducers_EcalRecalibRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalRecalibRecHitProducer_HH
/** \class EcalRecalibRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  \author Federico Ferri, University of Milano Bicocca and INFN
 *
 **/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"



class EcalRecalibRecHitProducer : public edm::EDProducer {

        public:
                explicit EcalRecalibRecHitProducer(const edm::ParameterSet& ps);
                ~EcalRecalibRecHitProducer();
                virtual void produce(edm::Event& evt, const edm::EventSetup& es);

        private:

		edm::InputTag EBRecHitCollection_;
		edm::InputTag EERecHitCollection_;
		edm::EDGetTokenT<EBRecHitCollection> EBRecHitToken_;
		edm::EDGetTokenT<EERecHitCollection> EERecHitToken_;
		  
                std::string EBRecalibRecHitCollection_; // secondary name to be given to EB collection of hits
                std::string EERecalibRecHitCollection_; // secondary name to be given to EE collection of hits

                bool doEnergyScale_;
                bool doIntercalib_;
                bool doLaserCorrections_;
		bool doEnergyScaleInverse_;
		bool doIntercalibInverse_;
                bool doLaserCorrectionsInverse_;

                EcalRecHitAbsAlgo* EBalgo_;
                EcalRecHitAbsAlgo* EEalgo_;
};
#endif
