#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerGlobal_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerGlobal_hh

/** \class EcalUncalibRecHitRecGlobalAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  $Id: EcalUncalibRecHitWorkerGlobal.h,v 1.7 2010/03/10 15:42:49 franzoni Exp $
  *  $Date: 2010/03/10 15:42:49 $
  *  $Revision: 1.7 $
  *  \author R. Bruneliere - A. Zabi
  */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecWeightsAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecChi2Algo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRatioMethodAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitLeadingEdgeAlgo.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EBShape.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EEShape.h"


namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
}

class EcalUncalibRecHitWorkerGlobal : public EcalUncalibRecHitWorkerBaseClass {

        public:
                EcalUncalibRecHitWorkerGlobal(const edm::ParameterSet&);
                virtual ~EcalUncalibRecHitWorkerGlobal() {};

                void set(const edm::EventSetup& es);
                bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result);

        protected:

                double pedVec[3];
		double pedRMSVec[3];
                double gainRatios[3];

                edm::ESHandle<EcalPedestals> peds;
                edm::ESHandle<EcalGainRatios>  gains;

                template < class C > int isSaturated(const C & digi);

                // weights method
                edm::ESHandle<EcalWeightXtalGroups>  grps;
                edm::ESHandle<EcalTBWeights> wgts;
                const EcalWeightSet::EcalWeightMatrix* weights[2];
                const EcalWeightSet::EcalChi2WeightMatrix* chi2mat[2];
                EcalUncalibRecHitRecWeightsAlgo<EBDataFrame> weightsMethod_barrel_;
                EcalUncalibRecHitRecWeightsAlgo<EEDataFrame> weightsMethod_endcap_;
                const EEShape testbeamEEShape; // used in the chi2
                const EBShape testbeamEBShape; // can be replaced by simple shape arrays of float in the future

                // ratio method
                std::vector<double> EBtimeFitParameters_; 
                std::vector<double> EEtimeFitParameters_; 
                std::vector<double> EBamplitudeFitParameters_; 
                std::vector<double> EEamplitudeFitParameters_; 
                std::pair<double,double> EBtimeFitLimits_;  
                std::pair<double,double> EEtimeFitLimits_;  
                EcalUncalibRecHitRatioMethodAlgo<EBDataFrame> ratioMethod_barrel_;
                EcalUncalibRecHitRatioMethodAlgo<EEDataFrame> ratioMethod_endcap_;

                double EBtimeConstantTerm_;
                double EBtimeNconst_;
                double EEtimeConstantTerm_;
                double EEtimeNconst_;
                double outOfTimeThreshEB_;
                double outOfTimeThreshEE_;
                double amplitudeThreshEB_;
                double amplitudeThreshEE_;
                double ebSpikeThresh_;

                // leading edge method
                edm::ESHandle<EcalTimeCalibConstants> itime;
                std::vector<double> ebPulseShape_;
                std::vector<double> eePulseShape_;
                EcalUncalibRecHitLeadingEdgeAlgo<EBDataFrame> leadingEdgeMethod_barrel_;
                EcalUncalibRecHitLeadingEdgeAlgo<EEDataFrame> leadingEdgeMethod_endcap_;

};

#endif
