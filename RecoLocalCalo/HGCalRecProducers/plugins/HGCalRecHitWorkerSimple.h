#ifndef RecoLocalCalo_HGCalRecProducers_HGCalRecHitWorkerSimple_hh
#define RecoLocalCalo_HGCalRecProducers_HGCalRecHitWorkerSimple_hh

/** \class HGCalRecHitSimpleAlgo
  *  Simple algoritm to make HGCAL rechits from HGCAL uncalibrated rechits
  *
  *  \author Valeri Andreev
  */

#include "RecoLocalCalo/HGCalRecProducers/interface/HGCalRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/HGCalRecAlgos/interface/HGCalRecHitSimpleAlgo.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "FWCore/Framework/interface/ESHandle.h"

class HGCalRecHitWorkerSimple : public HGCalRecHitWorkerBaseClass {
        public:
                HGCalRecHitWorkerSimple(const edm::ParameterSet&);
                virtual ~HGCalRecHitWorkerSimple();                       
        
                void set(const edm::EventSetup& es);
                bool run(const edm::Event& evt, const HGCUncalibratedRecHit& uncalibRH, HGCRecHitCollection & result);

        protected:

		double HGCEEmipInKeV_, HGCEElsbInMIP_, HGCEEmip2noise_;
		double HGCHEFmipInKeV_, HGCHEFlsbInMIP_, HGCHEFmip2noise_;
		double HGCHEBmipInKeV_, HGCHEBlsbInMIP_, HGCHEBmip2noise_;
		double hgceeADCtoGeV_, hgchefADCtoGeV_, hgchebADCtoGeV_;

                std::vector<int> v_chstatus_;

		std::vector<int> v_DB_reco_flags_;

                bool killDeadChannels_;
 
                HGCalRecHitSimpleAlgo * rechitMaker_;
};

#endif
