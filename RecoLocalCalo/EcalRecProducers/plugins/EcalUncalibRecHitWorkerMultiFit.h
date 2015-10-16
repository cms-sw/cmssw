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
#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"
#include "CondFormats/EcalObjects/interface/EcalTimeOffsetConstant.h"
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include "CondFormats/EcalObjects/interface/EcalTimeBiasCorrections.h"
#include "CondFormats/EcalObjects/interface/EcalSamplesCorrelation.h"
#include "CondFormats/EcalObjects/interface/EcalPulseShapes.h"
#include "CondFormats/EcalObjects/interface/EcalPulseCovariances.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EigenMatrixTypes.h"


namespace edm {
        class Event;
        class EventSetup;
        class ParameterSet;
        class ParameterSetDescription;
}

class EcalUncalibRecHitWorkerMultiFit : public EcalUncalibRecHitWorkerBaseClass {

        public:
                EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&, edm::ConsumesCollector& c);
		EcalUncalibRecHitWorkerMultiFit() {};
                virtual ~EcalUncalibRecHitWorkerMultiFit() {};

                void set(const edm::EventSetup& es) override;
                void set(const edm::Event& evt) override;
                bool run(const edm::Event& evt, const EcalDigiCollection::const_iterator & digi, EcalUncalibratedRecHitCollection & result) override;
		
		edm::ParameterSetDescription getAlgoDescription();
        protected:

                double pedVec[3];
		double pedRMSVec[3];
                double gainRatios[3];

                edm::ESHandle<EcalPedestals> peds;
                edm::ESHandle<EcalGainRatios>  gains;
                edm::ESHandle<EcalSamplesCorrelation> noisecovariances;
                edm::ESHandle<EcalPulseShapes> pulseshapes;
                edm::ESHandle<EcalPulseCovariances> pulsecovariances;

                double timeCorrection(float ampli,
                    const std::vector<float>& amplitudeBins, const std::vector<float>& shiftBins);

                const SampleMatrix &noisecor(bool barrel, int gain) const;                
                
                // multifit method
                SampleMatrix noisecorEBg12;
                SampleMatrix noisecorEEg12;
                SampleMatrix noisecorEBg6;
                SampleMatrix noisecorEEg6;
                SampleMatrix noisecorEBg1;
                SampleMatrix noisecorEEg1;
                FullSampleVector fullpulseEB;
                FullSampleVector fullpulseEE;
                FullSampleMatrix fullpulsecovEB;
                FullSampleMatrix fullpulsecovEE;
                BXVector activeBX;
                bool ampErrorCalculation_;
                bool useLumiInfoRunHeader_;
                EcalUncalibRecHitMultiFitAlgo multiFitMethod_;
                
		int bunchSpacingManual_;
                edm::EDGetTokenT<unsigned int> bunchSpacing_; 

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
                bool doPrefitEB_;
                bool doPrefitEE_;
		double prefitMaxChiSqEB_;
		double prefitMaxChiSqEE_;

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
                double EEtimeConstantTerm_;
                double EBtimeNconst_;
                double EEtimeNconst_;
                double outOfTimeThreshG12pEB_;
                double outOfTimeThreshG12mEB_;
                double outOfTimeThreshG61pEB_;
                double outOfTimeThreshG61mEB_;
                double outOfTimeThreshG12pEE_;
                double outOfTimeThreshG12mEE_;
                double outOfTimeThreshG61pEE_;
                double outOfTimeThreshG61mEE_;
                double amplitudeThreshEB_;
                double amplitudeThreshEE_;
                double ebSpikeThresh_;

                edm::ESHandle<EcalTimeBiasCorrections> timeCorrBias_;

                edm::ESHandle<EcalTimeCalibConstants> itime;
		edm::ESHandle<EcalTimeOffsetConstant> offtime;
                std::vector<double> ebPulseShape_;
                std::vector<double> eePulseShape_;


                // chi2 thresholds for flags settings
                bool kPoorRecoFlagEB_;
                bool kPoorRecoFlagEE_;
                double chi2ThreshEB_;
                double chi2ThreshEE_;


};

#endif
