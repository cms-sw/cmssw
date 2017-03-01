#ifndef RecoLocalCalo_EcalRecProducers_EcalRecHitProducer_hh
#define RecoLocalCalo_EcalRecProducers_EcalRecHitProducer_hh

/** \class EcalRecHitProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
 *
 **/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"

//#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitAbsAlgo.h"

#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerBaseClass.h"

class EcalCleaningAlgo;
class EBDetId;
class EEDetId;
class EcalTrigTowerDetId;
class EcalScDetId;

class EcalRecHitProducer : public edm::stream::EDProducer<> {

        public:
                explicit EcalRecHitProducer(const edm::ParameterSet& ps);
                ~EcalRecHitProducer();
                virtual void produce(edm::Event& evt, const edm::EventSetup& es) override;
		static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);
        private:

                std::string ebRechitCollection_; // secondary name to be given to EB collection of hits
                std::string eeRechitCollection_; // secondary name to be given to EE collection of hits

                bool recoverEBIsolatedChannels_;
                bool recoverEEIsolatedChannels_;
                bool recoverEBVFE_;
                bool recoverEEVFE_;
                bool recoverEBFE_;
                bool recoverEEFE_;
                bool killDeadChannels_;


                EcalRecHitWorkerBaseClass * worker_;
                EcalRecHitWorkerBaseClass * workerRecover_;

		EcalCleaningAlgo * cleaningAlgo_;  
		
		edm::EDGetTokenT<EBUncalibratedRecHitCollection> ebUncalibRecHitToken_;
		edm::EDGetTokenT<EEUncalibratedRecHitCollection> eeUncalibRecHitToken_;
		edm::EDGetTokenT<std::set<EBDetId> > ebDetIdToBeRecoveredToken_; 
		edm::EDGetTokenT<std::set<EEDetId> > eeDetIdToBeRecoveredToken_;
		edm::EDGetTokenT<std::set<EcalTrigTowerDetId> > ebFEToBeRecoveredToken_;
		edm::EDGetTokenT< std::set<EcalScDetId> >  eeFEToBeRecoveredToken_;

};

#endif
