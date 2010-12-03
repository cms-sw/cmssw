#ifndef RecoLocalCalo_EcalRecProducers_EcalRecHitWorkerRecover_hh
#define RecoLocalCalo_EcalRecProducers_EcalRecHitWorkerRecover_hh

/** \class EcalRecHitWorkerRecover
  *  Algorithms to recover dead channels
  *
  *  $Id: EcalRecHitWorkerRecover.h,v 1.7 2010/12/03 12:58:16 argiro Exp $
  *  $Date: 2010/12/03 12:58:16 $
  *  $Revision: 1.7 $
  */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"

#include "FWCore/Framework/interface/ESHandle.h"


#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
#include "Geometry/CaloGeometry/interface/CaloGeometry.h"
#include "Geometry/CaloTopology/interface/EcalTrigTowerConstituentsMap.h"
#include "Geometry/CaloGeometry/interface/CaloSubdetectorGeometry.h"
#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"

#include "CalibCalorimetry/EcalTPGTools/interface/EcalTPGScale.h"
#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"

class EcalRecHitWorkerRecover : public EcalRecHitWorkerBaseClass {
        public: 
                EcalRecHitWorkerRecover(const edm::ParameterSet&);
                virtual ~EcalRecHitWorkerRecover() {};

                void set(const edm::EventSetup& es);
                bool run(const edm::Event& evt, const EcalUncalibratedRecHit& uncalibRH, EcalRecHitCollection & result);

        protected:

                void insertRecHit( const EcalRecHit &hit, EcalRecHitCollection &collection );
		float recCheckCalib(float energy, int ieta);
                bool  alreadyInserted( const DetId & id );
		float estimateEnergy(int ieta, EcalRecHitCollection* hits, 
				     std::set<DetId> sId, 
				     std::vector<DetId> vId);

                edm::ESHandle<EcalLaserDbService> laser;

                // isolated dead channels
                edm::ESHandle<CaloTopology> caloTopology_;
		edm::ESHandle<CaloGeometry> caloGeometry_;
                double singleRecoveryThreshold_;
                std::string singleRecoveryMethod_;
                bool killDeadChannels_;

                bool recoverEBIsolatedChannels_;
                bool recoverEEIsolatedChannels_;
                bool recoverEBVFE_;
                bool recoverEEVFE_;
                bool recoverEBFE_;
                bool recoverEEFE_;

                // dead FE
                EcalTPGScale ecalScale_;
                edm::InputTag tpDigiCollection_;
                edm::ESHandle< EcalElectronicsMapping > pEcalMapping_;
                const EcalElectronicsMapping *ecalMapping_;
		double logWarningEtThreshold_EB_FE_;
		double logWarningEtThreshold_EE_FE_;

                edm::ESHandle<EcalTrigTowerConstituentsMap> ttMap_;
 
                edm::ESHandle<CaloSubdetectorGeometry> pEBGeom_;
                edm::ESHandle<CaloSubdetectorGeometry> pEEGeom_;
                const CaloSubdetectorGeometry * ebGeom_;
                const CaloSubdetectorGeometry * eeGeom_;
		const CaloGeometry* geo_;

                EcalRecHitSimpleAlgo * rechitMaker_;

                std::set<DetId> recoveredDetIds_EB_;
                std::set<DetId> recoveredDetIds_EE_;
};

#endif
