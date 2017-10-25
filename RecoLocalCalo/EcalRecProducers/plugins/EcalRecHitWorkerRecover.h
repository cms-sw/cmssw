#ifndef RecoLocalCalo_EcalRecProducers_EcalRecHitWorkerRecover_hh
#define RecoLocalCalo_EcalRecProducers_EcalRecHitWorkerRecover_hh

/** \class EcalRecHitWorkerRecover
  *  Algorithms to recover dead channels
  *
  */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include <vector>

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"
#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "DataFormats/EcalDigi/interface/EcalDigiCollections.h"

#include "RecoLocalCalo/EcalDeadChannelRecoveryAlgos/interface/EcalDeadChannelRecoveryAlgos.h"

class EcalRecHitWorkerRecover : public EcalRecHitWorkerBaseClass {
        public: 
                EcalRecHitWorkerRecover(const edm::ParameterSet&, edm::ConsumesCollector& c);
                ~EcalRecHitWorkerRecover() override {};

                void set(const edm::EventSetup& es) override;
                bool run(const edm::Event& evt, const EcalUncalibratedRecHit& uncalibRH, EcalRecHitCollection & result) override;

        protected:

                void insertRecHit( const EcalRecHit &hit, EcalRecHitCollection &collection );
		float recCheckCalib(float energy, int ieta);
                bool  alreadyInserted( const DetId & id );
		float estimateEnergy(int ieta, EcalRecHitCollection* hits, 
				     const std::set<DetId>& sId, 
				     const std::vector<DetId>& vId);
		bool checkChannelStatus(const DetId& id,
					const std::vector<int>& statusestoexclude);

                edm::ESHandle<EcalLaserDbService> laser;

                // isolated dead channels
                edm::ESHandle<CaloTopology>      caloTopology_;
		edm::ESHandle<CaloGeometry>      caloGeometry_;
		edm::ESHandle<EcalChannelStatus> chStatus_;
		

                double singleRecoveryThreshold_;
                std::string singleRecoveryMethod_;
                bool killDeadChannels_;

                bool recoverEBIsolatedChannels_;
                bool recoverEEIsolatedChannels_;
                bool recoverEBVFE_;
                bool recoverEEVFE_;
                bool recoverEBFE_;
                bool recoverEEFE_;
		
		// list of channel statuses for which recovery in EE should 
                // not be attempted 
		std::vector<int> dbStatusToBeExcludedEE_;
		std::vector<int> dbStatusToBeExcludedEB_;

                // dead FE
                EcalTPGScale ecalScale_;
                edm::EDGetTokenT<EcalTrigPrimDigiCollection> tpDigiToken_;
                edm::ESHandle< EcalElectronicsMapping > pEcalMapping_;
                const EcalElectronicsMapping *ecalMapping_;
		double logWarningEtThreshold_EB_FE_;
		double logWarningEtThreshold_EE_FE_;

                edm::ESHandle<EcalTrigTowerConstituentsMap> ttMap_;
 
                edm::ESHandle<CaloSubdetectorGeometry> pEBGeom_;
                const CaloSubdetectorGeometry * ebGeom_;
		const CaloGeometry* geo_;

                std::unique_ptr<EcalRecHitSimpleAlgo> rechitMaker_;

                std::set<DetId> recoveredDetIds_EB_;
                std::set<DetId> recoveredDetIds_EE_;

		EcalTPGScale tpgscale_;

	EcalDeadChannelRecoveryAlgos<EBDetId> ebDeadChannelCorrector;
	EcalDeadChannelRecoveryAlgos<EEDetId> eeDeadChannelCorrector;

};

#endif
