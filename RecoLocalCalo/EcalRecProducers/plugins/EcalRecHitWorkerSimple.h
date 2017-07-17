#ifndef RecoLocalCalo_EcalRecProducers_EcalRecHitWorkerSimple_hh
#define RecoLocalCalo_EcalRecProducers_EcalRecHitWorkerSimple_hh

/** \class EcalRecHitSimpleAlgo
  *  Simple algoritm to make rechits from uncalibrated rechits
  *
  *  \author Shahram Rahatlou, University of Rome & INFN, March 2006
  */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalRecHitSimpleAlgo.h"

#include "FWCore/Framework/interface/ESHandle.h"

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"

#include "CalibCalorimetry/EcalLaserCorrection/interface/EcalLaserDbService.h"

class EcalRecHitWorkerSimple : public EcalRecHitWorkerBaseClass {
        public:
                EcalRecHitWorkerSimple(const edm::ParameterSet&, edm::ConsumesCollector& c);
				EcalRecHitWorkerSimple(const edm::ParameterSet&);
                virtual ~EcalRecHitWorkerSimple();                       
        
                void set(const edm::EventSetup& es);
                bool run(const edm::Event& evt, const EcalUncalibratedRecHit& uncalibRH, EcalRecHitCollection & result);



        protected:

		double EBLaserMIN_;
		double EELaserMIN_;
		double EBLaserMAX_;
		double EELaserMAX_;


        edm::ESHandle<EcalIntercalibConstants> ical;
        edm::ESHandle<EcalTimeCalibConstants> itime;
        edm::ESHandle<EcalTimeOffsetConstant> offtime;
        edm::ESHandle<EcalADCToGeVConstant> agc;
        edm::ESHandle<EcalChannelStatus> chStatus;
        std::vector<int> v_chstatus_;
        edm::ESHandle<EcalLaserDbService> laser;

		// Associate reco flagbit ( outer vector) to many db status flags (inner vector)
		std::vector<std::vector<uint32_t> > v_DB_reco_flags_;

		uint32_t setFlagBits(const std::vector<std::vector<uint32_t> >& map, 
				     const uint32_t& status  );

        uint32_t flagmask_; // do not propagate channels with these flags on

        bool killDeadChannels_;
        bool laserCorrection_;
        bool skipTimeCalib_;

        EcalRecHitSimpleAlgo * rechitMaker_;

};

#endif
