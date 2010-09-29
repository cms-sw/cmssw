#ifndef RecoLocalCalo_EcalRecProducers_EcalRecHitWorkerRecover_hh
#define RecoLocalCalo_EcalRecProducers_EcalRecHitWorkerRecover_hh

/** \class EcalRecHitSimpleAlgo
  *  Simple algoritm to make rechits from uncalibrated rechits
  *
  *  $Id: EcalRecHitWorkerRecover.h,v 1.4 2010/09/24 02:34:34 theofil Exp $
  *  $Date: 2010/09/24 02:34:34 $
  *  $Revision: 1.4 $
  *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
  */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"

#include "FWCore/Framework/interface/ESHandle.h"

//#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
//#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
//#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "Geometry/CaloTopology/interface/CaloTopology.h"
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
                bool alreadyInserted( const DetId & id );

                //edm::ESHandle<EcalIntercalibConstants> ical;
                //edm::ESHandle<EcalTimeCalibConstants> itime;
                //edm::ESHandle<EcalADCToGeVConstant> agc;
                //std::vector<int> v_chstatus_;
                //edm::ESHandle<EcalChannelStatus> chStatus;
                edm::ESHandle<EcalLaserDbService> laser;

                // isolated dead channels
                edm::ESHandle<CaloTopology> caloTopology_;
                double singleRecoveryThreshold_;
                std::string singleRecoveryMethod_;
                bool recoverDeadVFE_;
                bool killDeadChannels_;

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

                EcalRecHitSimpleAlgo * rechitMaker_;

                std::set<DetId> recoveredDetIds_EB_;
                std::set<DetId> recoveredDetIds_EE_;
};

#endif
