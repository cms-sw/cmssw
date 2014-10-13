#ifndef RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerGlobal_hh
#define RecoLocalCalo_EcalRecProducers_EcalUncalibRecHitRecWorkerGlobal_hh

/** \class EcalUncalibRecHitRecGlobalAlgo                                                                                                                                           
 *  Template used to compute amplitude, pedestal using a weights method                                                                                                            
 *                           time using a ratio method                                                                                                                             
 *                           chi2 using express method  
 *
 *  \author R. Bruneliere - A. Zabi
 */

#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerBaseClass.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitMultiFitAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitTimeWeightsAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecChi2Algo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRatioMethodAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitLeadingEdgeAlgo.h"
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes.h"


namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
}

class EcalUncalibRecHitWorkerMultiFit : public EcalUncalibRecHitWorkerBaseClass {

        public:
                EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&);
				//EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&);
                virtual ~EcalUncalibRecHitWorkerMultiFit() {};

                void set(const edm::EventSetup& es);
                bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result);

        protected:

                edm::ParameterSet EcalPulseShapeParameters_;

                double pedVec[3];
		double pedRMSVec[3];
                double gainRatios[3];

                edm::ESHandle<EcalPedestals> peds;
                edm::ESHandle<EcalGainRatios>  gains;

                double timeCorrectionEB(float ampliEB);
                double timeCorrectionEE(float ampliEE);
                double timeCorrectionEK(float ampliEE);

                const SampleMatrix &noisecor(bool barrel, int gain) const;                
                
                // multifit method
                SampleMatrix noisecorEBg12;
                SampleMatrix noisecorEEg12;
                SampleMatrix noisecorEKg12;
                SampleMatrix noisecorEBg6;
                SampleMatrix noisecorEEg6;
                SampleMatrix noisecorEKg6;
                SampleMatrix noisecorEBg1;
                SampleMatrix noisecorEEg1;
                SampleMatrix noisecorEKg1;
                FullSampleVector fullpulseEB;
                FullSampleVector fullpulseEE;
                FullSampleVector fullpulseEK;
                FullSampleMatrix fullpulsecovEB;
                FullSampleMatrix fullpulsecovEE;
                FullSampleMatrix fullpulsecovEK;
                BXVector activeBX;
                bool ampErrorCalculation_;
                EcalUncalibRecHitMultiFitAlgo multiFitMethod_;
                

                // determine which of the samples must actually be used by ECAL local reco
                edm::ESHandle<EcalSampleMask> sampleMaskHand_;                
                
                // time algorithm to be used to set the jitter and its uncertainty
                std::string timealgo_;

                // time weights method
                edm::ESHandle<EcalWeightXtalGroups>  grps;
                edm::ESHandle<EcalTBWeights> wgts;
                const EcalWeightSet::EcalWeightMatrix* weights[2];
                EcalUncalibRecHitTimeWeightsAlgo<EBDataFrame> weightsMethod_barrel_;
                EcalUncalibRecHitTimeWeightsAlgo<EEDataFrame> weightsMethod_endcap_;
                EcalUncalibRecHitTimeWeightsAlgo<EKDataFrame> weightsMethod_shashlik_;

                // ratio method
                std::vector<double> EBtimeFitParameters_; 
                std::vector<double> EEtimeFitParameters_; 
                std::vector<double> EKtimeFitParameters_; 
                std::vector<double> EBamplitudeFitParameters_; 
                std::vector<double> EEamplitudeFitParameters_; 
                std::vector<double> EKamplitudeFitParameters_; 
                std::pair<double,double> EBtimeFitLimits_;  
                std::pair<double,double> EEtimeFitLimits_;  
                std::pair<double,double> EKtimeFitLimits_;  
                bool                doEBtimeCorrection_;
                bool                doEEtimeCorrection_;
                bool                doEKtimeCorrection_;
                std::vector<double> EBtimeCorrAmplitudeBins_; 
                std::vector<double> EBtimeCorrShiftBins_; 
                std::vector<double> EEtimeCorrAmplitudeBins_; 
                std::vector<double> EEtimeCorrShiftBins_; 
                std::vector<double> EKtimeCorrAmplitudeBins_; 
                std::vector<double> EKtimeCorrShiftBins_; 
                EcalUncalibRecHitRatioMethodAlgo<EBDataFrame> ratioMethod_barrel_;
                EcalUncalibRecHitRatioMethodAlgo<EEDataFrame> ratioMethod_endcap_;
                EcalUncalibRecHitRatioMethodAlgo<EKDataFrame> ratioMethod_shashlik_;

                double EBtimeConstantTerm_;
                double EEtimeConstantTerm_;
                double EKtimeConstantTerm_;

                // leading edge method
                edm::ESHandle<EcalTimeCalibConstants> itime;
                edm::ESHandle<EcalTimeOffsetConstant> offtime;
                std::vector<double> ebPulseShape_;
                std::vector<double> eePulseShape_;
                std::vector<double> ekPulseShape_;
                EcalUncalibRecHitLeadingEdgeAlgo<EBDataFrame> leadingEdgeMethod_barrel_;
                EcalUncalibRecHitLeadingEdgeAlgo<EEDataFrame> leadingEdgeMethod_endcap_;
                EcalUncalibRecHitLeadingEdgeAlgo<EKDataFrame> leadingEdgeMethod_shashlik_;


                // chi2 thresholds for flags settings
                bool kPoorRecoFlagEB_;
                bool kPoorRecoFlagEE_;
                bool kPoorRecoFlagEK_;
                double chi2ThreshEB_;
                double chi2ThreshEE_;
                double chi2ThreshEK_;


 private:
                void fillInputs(const edm::ParameterSet& params);

};

#endif
