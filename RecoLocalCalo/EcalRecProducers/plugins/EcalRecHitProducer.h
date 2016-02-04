#ifndef RecoLocalCalo_EcalRecProducers_EcalRecHitProducer_hh
#define RecoLocalCalo_EcalRecProducers_EcalRecHitProducer_hh

/** \class EcalRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  $Id: EcalRecHitProducer.h,v 1.4 2011/01/12 13:59:24 argiro Exp $
 *  $Date: 2011/01/12 13:59:24 $
 *  $Revision: 1.4 $
 *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
 *
 **/

#include "FWCore/Framework/interface/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerBaseClass.h"

class EcalCleaningAlgo;

class EcalRecHitProducer : public edm::EDProducer {

        public:
                explicit EcalRecHitProducer(const edm::ParameterSet& ps);
                ~EcalRecHitProducer();
                virtual void produce(edm::Event& evt, const edm::EventSetup& es);

        private:

                edm::InputTag ebUncalibRecHitCollection_; // secondary name given to collection of EB uncalib rechits
                edm::InputTag eeUncalibRecHitCollection_; // secondary name given to collection of EE uncalib rechits
                std::string ebRechitCollection_; // secondary name to be given to EB collection of hits
                std::string eeRechitCollection_; // secondary name to be given to EE collection of hits

                bool recoverEBIsolatedChannels_;
                bool recoverEEIsolatedChannels_;
                bool recoverEBVFE_;
                bool recoverEEVFE_;
                bool recoverEBFE_;
                bool recoverEEFE_;
                bool killDeadChannels_;

                edm::InputTag ebDetIdToBeRecovered_;
                edm::InputTag eeDetIdToBeRecovered_;
                edm::InputTag ebFEToBeRecovered_;
                edm::InputTag eeFEToBeRecovered_;

                EcalRecHitWorkerBaseClass * worker_;
                EcalRecHitWorkerBaseClass * workerRecover_;

		EcalCleaningAlgo * cleaningAlgo_;  
};

#endif
