#ifndef RecoLocalCalo_EcalRecProducers_EcalDetIdToBeRecoveredProducer_hh
#define RecoLocalCalo_EcalRecProducers_EcalDetIdToBeRecoveredProducer_hh

/** \class EcalDetIdToBeRecoveredProducer
 *   produce ECAL rechits from uncalibrated rechits
 *
 *  $Id:
 *  $Date:
 *  $Revision:
 *  \author Federico Ferri, CEA-Saclay IRFU/SPP
 *
 **/

#include "FWCore/Framework/interface/stream/EDProducer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/ESHandle.h"

#include "DataFormats/EcalDigi/interface/EBSrFlag.h"
#include "DataFormats/EcalDigi/interface/EESrFlag.h"

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"


class EcalDetIdToBeRecoveredProducer : public edm::stream::EDProducer<> {

        public:
                explicit EcalDetIdToBeRecoveredProducer(const edm::ParameterSet& ps);
                ~EcalDetIdToBeRecoveredProducer();
                virtual void produce(edm::Event& evt, const edm::EventSetup& es) override final;
                virtual void beginRun(edm::Run const& run, const edm::EventSetup& es) override final;

        private:

                //edm::InputTag eeUncalibRecHitCollection_; // secondary name given to collection of EE uncalib rechits
                //std::string eeRechitCollection_; // secondary name to be given to EE collection of hits

                const EcalChannelStatusMap * chStatus_;
                const EcalElectronicsMapping * ecalMapping_;
                edm::ESHandle<EcalTrigTowerConstituentsMap> ttMap_;

                /*
                 * InputTag for collections
                 */
                // SRP collections
		edm::EDGetTokenT<EBSrFlagCollection> ebSrFlagToken_; 
		edm::EDGetTokenT<EESrFlagCollection> eeSrFlagToken_; 

                // Integrity for xtal data
		edm::EDGetTokenT<EBDetIdCollection> ebIntegrityGainErrorsToken_; 
                edm::EDGetTokenT<EBDetIdCollection> ebIntegrityGainSwitchErrorsToken_;
                edm::EDGetTokenT<EBDetIdCollection> ebIntegrityChIdErrorsToken_;

                // Integrity for xtal data - EE specific (to be rivisited towards EB+EE common collection)
                edm::EDGetTokenT<EEDetIdCollection> eeIntegrityGainErrorsToken_;
                edm::EDGetTokenT<EEDetIdCollection> eeIntegrityGainSwitchErrorsToken_;
                edm::EDGetTokenT<EEDetIdCollection> eeIntegrityChIdErrorsToken_;

                // Integrity Errors
                edm::EDGetTokenT<EcalElectronicsIdCollection> integrityTTIdErrorsToken_;
                edm::EDGetTokenT<EcalElectronicsIdCollection> integrityBlockSizeErrorsToken_;

                /*
                 * output collections
                 */
                std::string ebDetIdCollection_;
                std::string eeDetIdCollection_;
                std::string ttDetIdCollection_;
                std::string scDetIdCollection_;
};

#endif
