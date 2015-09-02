#ifndef RecoLocalCalo_EcalRecProducers_EcalRecalibRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalRecalibRecHitProducer_HH
/** \class EcalRecalibRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  \author Federico Ferri, University of Milano Bicocca and INFN
 *
 **/

#include "FWCore/Framework/interface/global/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"



class EcalRecalibRecHitProducer : public edm::global::EDProducer<> {

        public:
                explicit EcalRecalibRecHitProducer(const edm::ParameterSet& ps);
                virtual void produce(edm::StreamID sid, edm::Event& evt, const edm::EventSetup& es) const override;

        private:
		const edm::InputTag EBRecHitCollection_;
		const edm::InputTag EERecHitCollection_;
		const edm::EDGetTokenT<EBRecHitCollection> EBRecHitToken_;
		const edm::EDGetTokenT<EERecHitCollection> EERecHitToken_;
		  
                const std::string EBRecalibRecHitCollection_; // secondary name to be given to EB collection of hits
                const std::string EERecalibRecHitCollection_; // secondary name to be given to EE collection of hits

                const bool doEnergyScale_;
                const bool doIntercalib_;
                const bool doLaserCorrections_;
		const bool doEnergyScaleInverse_;
		const bool doIntercalibInverse_;
                const bool doLaserCorrectionsInverse_;
};
#endif
