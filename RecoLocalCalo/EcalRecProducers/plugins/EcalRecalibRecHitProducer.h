#ifndef RecoLocalCalo_EcalRecProducers_EcalRecalibRecHitProducer_HH
#define RecoLocalCalo_EcalRecProducers_EcalRecalibRecHitProducer_HH
/** \class EcalRecalibRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  $Id: EcalRecalibRecHitProducer.h,v 1.3 2012/03/06 23:53:34 ferriff Exp $
 *  $Date: 2012/03/06 23:53:34 $
 *  $Revision: 1.3 $
 *  \author Federico Ferri, University of Milano Bicocca and INFN
 *
 **/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"

// forward declaration
class EcalRecalibRecHitProducer : public edm::EDProducer {

        public:
                explicit EcalRecalibRecHitProducer(const edm::ParameterSet& ps);
                ~EcalRecalibRecHitProducer();
                virtual void produce(edm::Event& evt, const edm::EventSetup& es);

        private:

                edm::InputTag EBRecHitCollection_; // secondary name given to collection of EB uncalib rechits
                edm::InputTag EERecHitCollection_; // secondary name given to collection of EE uncalib rechits
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
