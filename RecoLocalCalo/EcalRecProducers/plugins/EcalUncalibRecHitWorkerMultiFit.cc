#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerMultiFit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeBiasCorrectionsRcd.h"
#include "CondFormats/DataRecord/interface/EcalSamplesCorrelationRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseShapesRcd.h"
#include "CondFormats/DataRecord/interface/EcalPulseCovariancesRcd.h"

#include <FWCore/ParameterSet/interface/ConfigurationDescriptions.h>
#include <FWCore/ParameterSet/interface/ParameterSetDescription.h>
#include <FWCore/ParameterSet/interface/EmptyGroupDescription.h>

EcalUncalibRecHitWorkerMultiFit::EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&ps,edm::ConsumesCollector& c) :
  EcalUncalibRecHitWorkerBaseClass(ps,c),
  noisecorEBg12(SampleMatrix::Zero()), noisecorEEg12(SampleMatrix::Zero()),
  noisecorEBg6(SampleMatrix::Zero()), noisecorEEg6(SampleMatrix::Zero()),
  noisecorEBg1(SampleMatrix::Zero()), noisecorEEg1(SampleMatrix::Zero()),
  fullpulseEB(FullSampleVector::Zero()),fullpulseEE(FullSampleVector::Zero()),
  fullpulsecovEB(FullSampleMatrix::Zero()),fullpulsecovEE(FullSampleMatrix::Zero()) {

  // get the BX for the pulses to be activated
  std::vector<int32_t> activeBXs = ps.getParameter< std::vector<int32_t> >("activeBXs");
  activeBX.resize(activeBXs.size());
  for (unsigned int ibx=0; ibx<activeBXs.size(); ++ibx) {
    activeBX.coeffRef(ibx) = activeBXs[ibx];
  }

  // uncertainty calculation (CPU intensive)
  ampErrorCalculation_ = ps.getParameter<bool>("ampErrorCalculation");
  useLumiInfoRunHeader_ = ps.getParameter<bool>("useLumiInfoRunHeader");
  
  if (useLumiInfoRunHeader_) {
    bunchSpacing_ = c.consumes<int>(edm::InputTag("addPileupInfo","bunchSpacing"));
    bunchSpacingManual_ = 0;
  } else {
    bunchSpacingManual_ = ps.getParameter<int>("bunchSpacing");
  }

  doPrefitEB_ = ps.getParameter<bool>("doPrefitEB");
  doPrefitEE_ = ps.getParameter<bool>("doPrefitEE");

  prefitMaxChiSqEB_ = ps.getParameter<double>("prefitMaxChiSqEB");
  prefitMaxChiSqEE_ = ps.getParameter<double>("prefitMaxChiSqEE");

  // algorithm to be used for timing
  timealgo_ = ps.getParameter<std::string>("timealgo");
  
  // ratio method parameters
  EBtimeFitParameters_ = ps.getParameter<std::vector<double> >("EBtimeFitParameters"); 
  EEtimeFitParameters_ = ps.getParameter<std::vector<double> >("EEtimeFitParameters"); 
  EBamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EBamplitudeFitParameters");
  EEamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EEamplitudeFitParameters");
  EBtimeFitLimits_.first  = ps.getParameter<double>("EBtimeFitLimits_Lower");
  EBtimeFitLimits_.second = ps.getParameter<double>("EBtimeFitLimits_Upper");
  EEtimeFitLimits_.first  = ps.getParameter<double>("EEtimeFitLimits_Lower");
  EEtimeFitLimits_.second = ps.getParameter<double>("EEtimeFitLimits_Upper");
  EBtimeConstantTerm_=ps.getParameter<double>("EBtimeConstantTerm");
  EEtimeConstantTerm_=ps.getParameter<double>("EEtimeConstantTerm");
  EBtimeNconst_=ps.getParameter<double>("EBtimeNconst");
  EEtimeNconst_=ps.getParameter<double>("EEtimeNconst");
  outOfTimeThreshG12pEB_ = ps.getParameter<double>("outOfTimeThresholdGain12pEB");
  outOfTimeThreshG12mEB_ = ps.getParameter<double>("outOfTimeThresholdGain12mEB");
  outOfTimeThreshG61pEB_ = ps.getParameter<double>("outOfTimeThresholdGain61pEB");
  outOfTimeThreshG61mEB_ = ps.getParameter<double>("outOfTimeThresholdGain61mEB");
  outOfTimeThreshG12pEE_ = ps.getParameter<double>("outOfTimeThresholdGain12pEE");
  outOfTimeThreshG12mEE_ = ps.getParameter<double>("outOfTimeThresholdGain12mEE");
  outOfTimeThreshG61pEE_ = ps.getParameter<double>("outOfTimeThresholdGain61pEE");
  outOfTimeThreshG61mEE_ = ps.getParameter<double>("outOfTimeThresholdGain61mEE");
  amplitudeThreshEB_ = ps.getParameter<double>("amplitudeThresholdEB");
  amplitudeThreshEE_ = ps.getParameter<double>("amplitudeThresholdEE");

  // spike threshold
  ebSpikeThresh_ = ps.getParameter<double>("ebSpikeThreshold");

  ebPulseShape_ = ps.getParameter<std::vector<double> >("ebPulseShape");
  eePulseShape_ = ps.getParameter<std::vector<double> >("eePulseShape");

  // chi2 parameters for flags determination
  kPoorRecoFlagEB_ = ps.getParameter<bool>("kPoorRecoFlagEB");
  kPoorRecoFlagEE_ = ps.getParameter<bool>("kPoorRecoFlagEE");;
  chi2ThreshEB_=ps.getParameter<double>("chi2ThreshEB_");
  chi2ThreshEE_=ps.getParameter<double>("chi2ThreshEE_");

}



void
EcalUncalibRecHitWorkerMultiFit::set(const edm::EventSetup& es)
{

        // common setup
        es.get<EcalGainRatiosRcd>().get(gains);
        es.get<EcalPedestalsRcd>().get(peds);

        // for the multifit method
        if(!ampErrorCalculation_) multiFitMethod_.disableErrorCalculation();
        es.get<EcalSamplesCorrelationRcd>().get(noisecovariances);
        es.get<EcalPulseShapesRcd>().get(pulseshapes);
        es.get<EcalPulseCovariancesRcd>().get(pulsecovariances);

        // weights parameters for the time
        es.get<EcalWeightXtalGroupsRcd>().get(grps);
        es.get<EcalTBWeightsRcd>().get(wgts);

	// which of the samples need be used
	es.get<EcalSampleMaskRcd>().get(sampleMaskHand_);

        // for the ratio method
        es.get<EcalTimeCalibConstantsRcd>().get(itime);
        es.get<EcalTimeOffsetConstantRcd>().get(offtime);

        // for the time correction methods
        es.get<EcalTimeBiasCorrectionsRcd>().get(timeCorrBias_);
}

void
EcalUncalibRecHitWorkerMultiFit::set(const edm::Event& evt)
{

  int bunchspacing = 450;

  if (useLumiInfoRunHeader_) {

    if (evt.isRealData()) {
      edm::RunNumber_t run = evt.run();
      if (run == 178003 ||
          run == 178004 ||
          run == 209089 ||
          run == 209106 ||
          run == 209109 ||
          run == 209146 ||
          run == 209148 ||
          run == 209151) {
        bunchspacing = 25;
      }
      else if (run < 253000) {
        bunchspacing = 50;
      }
      else {
	bunchspacing = 25;
      }
    }
    else {
      edm::Handle<int> bunchSpacingH;
      evt.getByToken(bunchSpacing_,bunchSpacingH);
      bunchspacing = *bunchSpacingH;
    }
    
  }
  else {
    bunchspacing = bunchSpacingManual_;
  }

  if (useLumiInfoRunHeader_ || bunchSpacingManual_ > 0){
    if (bunchspacing == 25) {
      activeBX.resize(10);
      activeBX << -5,-4,-3,-2,-1,0,1,2,3,4;
    }
    else {
      //50ns configuration otherwise (also for no pileup)
      activeBX.resize(5);
      activeBX << -4,-2,0,2,4;
    }
  }
 
}

/**
 * Amplitude-dependent time corrections; EE and EB have separate corrections:
 * EXtimeCorrAmplitudes (ADC) and EXtimeCorrShifts (ns) need to have the same number of elements
 * Bins must be ordered in amplitude. First-last bins take care of under-overflows.
 *
 * The algorithm is the same for EE and EB, only the correction vectors are different.
 *
 * @return Jitter (in clock cycles) which will be added to UncalibRechit.setJitter(), 0 if no correction is applied.
 */
double EcalUncalibRecHitWorkerMultiFit::timeCorrection(
    float ampli,
	const std::vector<float>& amplitudeBins,
    const std::vector<float>& shiftBins) {

  // computed initially in ns. Than turned in the BX's, as
  // EcalUncalibratedRecHit need be.
  double theCorrection = 0;

  // sanity check for arrays
  if (amplitudeBins.size() == 0) {
    edm::LogError("EcalRecHitError")
        << "timeCorrAmplitudeBins is empty, forcing no time bias corrections.";

    return 0;
  }

  if (amplitudeBins.size() != shiftBins.size()) {
    edm::LogError("EcalRecHitError")
        << "Size of timeCorrAmplitudeBins different from "
           "timeCorrShiftBins. Forcing no time bias corrections. ";

    return 0;
  }

  int myBin = -1;
  for (int bin = 0; bin < (int) amplitudeBins.size(); bin++) {
    if (ampli > amplitudeBins.at(bin)) {
      myBin = bin;
    } else {
      break;
	}
  }

  if (myBin == -1) {
    theCorrection = shiftBins.at(0);
  } else if (myBin == ((int)(amplitudeBins.size() - 1))) {
    theCorrection = shiftBins.at(myBin);
  } else if (-1 < myBin && myBin < ((int) amplitudeBins.size() - 1)) {
    // interpolate linearly between two assingned points
    theCorrection = (shiftBins.at(myBin + 1) - shiftBins.at(myBin));
    theCorrection *= (((double) ampli) - amplitudeBins.at(myBin)) /
                     (amplitudeBins.at(myBin + 1) - amplitudeBins.at(myBin));
    theCorrection += shiftBins.at(myBin);
  } else {
    edm::LogError("EcalRecHitError")
        << "Assigning time correction impossible. Setting it to 0 ";
    theCorrection = 0.;
  }

  // convert ns into clocks
  return theCorrection / 25.;
}



bool
EcalUncalibRecHitWorkerMultiFit::run( const edm::Event & evt,
                const EcalDigiCollection::const_iterator & itdg,
                EcalUncalibratedRecHitCollection & result )
{
        DetId detid(itdg->id());

        const EcalSampleMask  *sampleMask_ = sampleMaskHand_.product();                
        
        // intelligence for recHit computation
        EcalUncalibratedRecHit uncalibRecHit;
        float offsetTime = 0;
        
        const EcalPedestals::Item * aped = 0;
        const EcalMGPAGainRatio * aGain = 0;
        const EcalXtalGroupId * gid = 0;
        const EcalPulseShapes::Item * aPulse = 0;
        const EcalPulseCovariances::Item * aPulseCov = 0;

        if (detid.subdetId()==EcalEndcap) {
                unsigned int hashedIndex = EEDetId(detid).hashedIndex();
                aped      = &peds->endcap(hashedIndex);
                aGain     = &gains->endcap(hashedIndex);
                gid       = &grps->endcap(hashedIndex);
                aPulse    = &pulseshapes->endcap(hashedIndex);
                aPulseCov = &pulsecovariances->endcap(hashedIndex);
                multiFitMethod_.setDoPrefit(doPrefitEE_);
		multiFitMethod_.setPrefitMaxChiSq(prefitMaxChiSqEE_);
		offsetTime = offtime->getEEValue();
        } else {
                unsigned int hashedIndex = EBDetId(detid).hashedIndex();
                aped      = &peds->barrel(hashedIndex);
                aGain     = &gains->barrel(hashedIndex);
                gid       = &grps->barrel(hashedIndex);
                aPulse    = &pulseshapes->barrel(hashedIndex);
                aPulseCov = &pulsecovariances->barrel(hashedIndex);
                multiFitMethod_.setDoPrefit(doPrefitEB_);
		multiFitMethod_.setPrefitMaxChiSq(prefitMaxChiSqEB_);
		offsetTime = offtime->getEBValue();
        }

        pedVec[0] = aped->mean_x12;
        pedVec[1] = aped->mean_x6;
        pedVec[2] = aped->mean_x1;
        pedRMSVec[0] = aped->rms_x12;
        pedRMSVec[1] = aped->rms_x6;
        pedRMSVec[2] = aped->rms_x1;
        gainRatios[0] = 1.;
        gainRatios[1] = aGain->gain12Over6();
        gainRatios[2] = aGain->gain6Over1()*aGain->gain12Over6();

        int nnoise = noisecovariances->EBG12SamplesCorrelation.size();
        for (int i=0; i<nnoise; ++i) {
          for (int j=0; j<nnoise; ++j) {
            int vidx = std::abs(j-i);
            noisecorEBg12(i,j) = noisecovariances->EBG12SamplesCorrelation[vidx];
            noisecorEEg12(i,j) = noisecovariances->EEG12SamplesCorrelation[vidx];
            noisecorEBg6(i,j)  = noisecovariances->EBG6SamplesCorrelation[vidx];
            noisecorEEg6(i,j)  = noisecovariances->EEG6SamplesCorrelation[vidx];
            noisecorEBg1(i,j)  = noisecovariances->EBG1SamplesCorrelation[vidx];
            noisecorEEg1(i,j)  = noisecovariances->EEG1SamplesCorrelation[vidx];        
          }
        }
        
        for (int i=0; i<EcalPulseShape::TEMPLATESAMPLES; ++i) {
          fullpulseEB(i+7) = aPulse->pdfval[i];
          fullpulseEE(i+7) = aPulse->pdfval[i];
        }

        for(int k=0; k<std::pow(EcalPulseShape::TEMPLATESAMPLES,2); ++k) {
          int i = k/EcalPulseShape::TEMPLATESAMPLES;
          int j = k%EcalPulseShape::TEMPLATESAMPLES;
          fullpulsecovEB(i+7,j+7) = aPulseCov->covval[i][j];
          fullpulsecovEE(i+7,j+7) = aPulseCov->covval[i][j];
        }
        
	// compute the right bin of the pulse shape using time calibration constants
	EcalTimeCalibConstantMap::const_iterator it = itime->find( detid );
	EcalTimeCalibConstant itimeconst = 0;
	if( it != itime->end() ) {
		  itimeconst = (*it);
	} else {
		  edm::LogError("EcalRecHitError") << "No time intercalib const found for xtal "
		  << detid.rawId()
		  << "! something wrong with EcalTimeCalibConstants in your DB? ";
	}

        // === amplitude computation ===
        int leadingSample = ((EcalDataFrame)(*itdg)).lastUnsaturatedSample();

        if ( leadingSample == 4 ) { // saturation on the expected max sample
	       uncalibRecHit = EcalUncalibratedRecHit( (*itdg).id(), 4095*12, 0, 0, 0);
               uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kSaturated );
	       // do not propagate the default chi2 = -1 value to the calib rechit (mapped to 64), set it to 0 when saturation
               uncalibRecHit.setChi2(0);
        } else if ( leadingSample >= 0 ) { // saturation on other samples: cannot extrapolate from the fourth one
               double pedestal = 0.;
               double gainratio = 1.;
               int gainId = ((EcalDataFrame)(*itdg)).sample(5).gainId();

               if (gainId==0 || gainId==3) {
                 pedestal = aped->mean_x1;
                 gainratio = aGain->gain6Over1()*aGain->gain12Over6();
               }
               else if (gainId==1) {
                 pedestal = aped->mean_x12;
                 gainratio = 1.;
               }
               else if (gainId==2) {
                 pedestal = aped->mean_x6;
                 gainratio = aGain->gain12Over6();
               }
               double amplitude = ((double)(((EcalDataFrame)(*itdg)).sample(5).adc()) - pedestal) * gainratio;
               uncalibRecHit = EcalUncalibratedRecHit( (*itdg).id(), amplitude, 0, 0, 0);
               uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kSaturated );
               // do not propagate the default chi2 = -1 value to the calib rechit (mapped to 64), set it to 0 when saturation
               uncalibRecHit.setChi2(0);
        } else {
                // multifit
                bool barrel = detid.subdetId()==EcalBarrel;
                int gain = 12;
                if (((EcalDataFrame)(*itdg)).hasSwitchToGain6()) {
                  gain = 6;
                }
                if (((EcalDataFrame)(*itdg)).hasSwitchToGain1()) {
                  gain = 1;
                }
                const SampleMatrix &noisecormat = noisecor(barrel,gain);
                const FullSampleVector &fullpulse = barrel ? fullpulseEB : fullpulseEE;
                const FullSampleMatrix &fullpulsecov = barrel ? fullpulsecovEB : fullpulsecovEE;
                                
                uncalibRecHit = multiFitMethod_.makeRecHit(*itdg, aped, aGain, noisecormat,fullpulse,fullpulsecov,activeBX);
                
                // === time computation ===
                if(timealgo_.compare("RatioMethod")==0) {
                  // ratio method
                  float const clockToNsConstant = 25.;
                  if (detid.subdetId()==EcalEndcap) {
                    ratioMethod_endcap_.init( *itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios );
                    ratioMethod_endcap_.computeTime( EEtimeFitParameters_, EEtimeFitLimits_, EEamplitudeFitParameters_ );
                    ratioMethod_endcap_.computeAmplitude( EEamplitudeFitParameters_);
                    EcalUncalibRecHitRatioMethodAlgo<EEDataFrame>::CalculatedRecHit crh = ratioMethod_endcap_.getCalculatedRecHit();
                    double theTimeCorrectionEE = timeCorrection(uncalibRecHit.amplitude(),
                                                                timeCorrBias_->EETimeCorrAmplitudeBins, timeCorrBias_->EETimeCorrShiftBins);
                    
                    uncalibRecHit.setJitter( crh.timeMax - 5 + theTimeCorrectionEE);
                    uncalibRecHit.setJitterError( std::sqrt(pow(crh.timeError,2) + std::pow(EEtimeConstantTerm_,2)/std::pow(clockToNsConstant,2)) );
                    
                    // consider flagging as kOutOfTime only if above noise
                    if (uncalibRecHit.amplitude() > pedRMSVec[0] * amplitudeThreshEE_){
                      float outOfTimeThreshP = outOfTimeThreshG12pEE_;
                      float outOfTimeThreshM = outOfTimeThreshG12mEE_;
                      // determine if gain has switched away from gainId==1 (x12 gain)
                      // and determine cuts (number of 'sigmas') to ose for kOutOfTime
                      // >3k ADC is necessasry condition for gain switch to occur
                      if (uncalibRecHit.amplitude() > 3000.){
                        for (int iSample = 0; iSample < EEDataFrame::MAXSAMPLES; iSample++) {
                          int GainId = ((EcalDataFrame)(*itdg)).sample(iSample).gainId();
                          if (GainId!=1) {
                            outOfTimeThreshP = outOfTimeThreshG61pEE_;
                            outOfTimeThreshM = outOfTimeThreshG61mEE_;
                            break;
                          }
                        }}
                      float correctedTime = (crh.timeMax-5) * clockToNsConstant + itimeconst + offsetTime;
                      float cterm         = EEtimeConstantTerm_;
                      float sigmaped      = pedRMSVec[0];  // approx for lower gains
                      float nterm         = EEtimeNconst_*sigmaped/uncalibRecHit.amplitude();
                      float sigmat        = std::sqrt( nterm*nterm  + cterm*cterm   );
                      if ( ( correctedTime > sigmat*outOfTimeThreshP )   ||
                           ( correctedTime < (-1.*sigmat*outOfTimeThreshM) )) 
                        {  uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kOutOfTime ); }
                    }

                  } else {
                    ratioMethod_barrel_.init( *itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios );
                    ratioMethod_barrel_.fixMGPAslew(*itdg);
                    ratioMethod_barrel_.computeTime( EBtimeFitParameters_, EBtimeFitLimits_, EBamplitudeFitParameters_ );
                    ratioMethod_barrel_.computeAmplitude( EBamplitudeFitParameters_);
                    EcalUncalibRecHitRatioMethodAlgo<EBDataFrame>::CalculatedRecHit crh = ratioMethod_barrel_.getCalculatedRecHit();
                    
                    double theTimeCorrectionEB = timeCorrection(uncalibRecHit.amplitude(),
                                                                timeCorrBias_->EBTimeCorrAmplitudeBins, timeCorrBias_->EBTimeCorrShiftBins);
                    
                    uncalibRecHit.setJitter( crh.timeMax - 5 + theTimeCorrectionEB);
                    uncalibRecHit.setJitterError( std::sqrt(std::pow(crh.timeError,2) + std::pow(EBtimeConstantTerm_,2)/std::pow(clockToNsConstant,2)) );

                    // consider flagging as kOutOfTime only if above noise
                    if (uncalibRecHit.amplitude() > pedRMSVec[0] * amplitudeThreshEB_){
                      float outOfTimeThreshP = outOfTimeThreshG12pEB_;
                      float outOfTimeThreshM = outOfTimeThreshG12mEB_;
                      // determine if gain has switched away from gainId==1 (x12 gain)
                      // and determine cuts (number of 'sigmas') to ose for kOutOfTime
                      // >3k ADC is necessasry condition for gain switch to occur
                      if (uncalibRecHit.amplitude() > 3000.){
                        for (int iSample = 0; iSample < EBDataFrame::MAXSAMPLES; iSample++) {
                          int GainId = ((EcalDataFrame)(*itdg)).sample(iSample).gainId();
                          if (GainId!=1) {
                            outOfTimeThreshP = outOfTimeThreshG61pEB_;
                            outOfTimeThreshM = outOfTimeThreshG61mEB_;
                            break;}
                        } }
                      float correctedTime = (crh.timeMax-5) * clockToNsConstant + itimeconst + offsetTime;
                      float cterm         = EBtimeConstantTerm_;
                      float sigmaped      = pedRMSVec[0];  // approx for lower gains
                      float nterm         = EBtimeNconst_*sigmaped/uncalibRecHit.amplitude();
                      float sigmat        = std::sqrt( nterm*nterm  + cterm*cterm   );
                      if ( ( correctedTime > sigmat*outOfTimeThreshP )   ||
                           ( correctedTime < (-1.*sigmat*outOfTimeThreshM) )) 
                        {   uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kOutOfTime );  }
                    }

                  }
                } else if(timealgo_.compare("WeightsMethod")==0) {
                  //  weights method on the PU subtracted pulse shape
                  std::vector<double> amplitudes;
                  for(unsigned int ibx=0; ibx<activeBX.size(); ++ibx) amplitudes.push_back(uncalibRecHit.outOfTimeAmplitude(ibx));
                  
                  EcalTBWeights::EcalTDCId tdcid(1);
                  EcalTBWeights::EcalTBWeightMap const & wgtsMap = wgts->getMap();
                  EcalTBWeights::EcalTBWeightMap::const_iterator wit;
                  wit = wgtsMap.find( std::make_pair(*gid,tdcid) );
                  if( wit == wgtsMap.end() ) {
                    edm::LogError("EcalUncalibRecHitError") << "No weights found for EcalGroupId: " 
                                                            << gid->id() << " and  EcalTDCId: " << tdcid
                                                            << "\n  skipping digi with id: " << detid.rawId();
                    
                    return false;
                  }
                  const EcalWeightSet& wset = wit->second; // this is the EcalWeightSet
                  
                  const EcalWeightSet::EcalWeightMatrix& mat1 = wset.getWeightsBeforeGainSwitch();
                  const EcalWeightSet::EcalWeightMatrix& mat2 = wset.getWeightsAfterGainSwitch();
                  
                  weights[0] = &mat1;
                  weights[1] = &mat2;
                  
                  double timerh;
                  if (detid.subdetId()==EcalEndcap) { 
                    timerh = weightsMethod_endcap_.time( *itdg, amplitudes, aped, aGain, fullpulse, weights);
                  } else {
                    timerh = weightsMethod_barrel_.time( *itdg, amplitudes, aped, aGain, fullpulse, weights);
                  }
                  uncalibRecHit.setJitter( timerh );
                  uncalibRecHit.setJitterError( 0. ); // not computed with weights
                }  else if(timealgo_.compare("None")==0) {
                  uncalibRecHit.setJitter( 0. );
                  uncalibRecHit.setJitterError( 0. );                  
                } else {
                  edm::LogError("EcalUncalibRecHitError") << "No time estimation algorithm called " 
                                                          << timealgo_
                                                          << "\n  setting jitter to 0. and jitter uncertainty to 10000. ";
                  
                  uncalibRecHit.setJitter( 0. );
                  uncalibRecHit.setJitterError( 10000. );
                }
        }

	// set flags if gain switch has occurred
	if( ((EcalDataFrame)(*itdg)).hasSwitchToGain6()  ) uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kHasSwitchToGain6 );
	if( ((EcalDataFrame)(*itdg)).hasSwitchToGain1()  ) uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kHasSwitchToGain1 );

        // set quality flags based on chi2 of the the fit
        /*
        if(detid.subdetId()==EcalEndcap) { 
          if(kPoorRecoFlagEE_ && uncalibRecHit.chi2()>chi2ThreshEE_) {
          bool samplesok = true;
          for (int sample =0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
            if (!sampleMask_->useSampleEE(sample)) {
              samplesok = false;
              break;
            }
          }
          if (samplesok) uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kPoorReco);
          }
        } else {
          if(kPoorRecoFlagEB_ && uncalibRecHit.chi2()>chi2ThreshEB_) {
          bool samplesok = true;
          for (int sample =0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
            if (!sampleMask_->useSampleEB(sample)) {
              samplesok = false;
              break;
            }
          }
          if (samplesok) uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kPoorReco);
          }
        }
        */

        // put the recHit in the collection
        if (detid.subdetId()==EcalEndcap) {
                result.push_back( uncalibRecHit );
        } else {
                result.push_back( uncalibRecHit );
        }

        return true;
}


const SampleMatrix &EcalUncalibRecHitWorkerMultiFit::noisecor(bool barrel, int gain) const {
  if (barrel) {
    if (gain==6) {
      return noisecorEBg6;
    }
    else if (gain==1) {
      return noisecorEBg1;
    }
    else {
      return noisecorEBg12;
    }    
  }
  else {
    if (gain==6) {
      return noisecorEEg6;
    }
    else if (gain==1) {
      return noisecorEEg1;
    }
    else {
      return noisecorEEg12;
    }        
  }
  
  return noisecorEBg12;
  
}

edm::ParameterSetDescription 
EcalUncalibRecHitWorkerMultiFit::getAlgoDescription() {
  
  edm::ParameterSetDescription psd0;
  psd0.addNode((edm::ParameterDescription<std::vector<double>>("EBPulseShapeTemplate", {1.13979e-02, 7.58151e-01, 1.00000e+00, 8.87744e-01, 6.73548e-01, 4.74332e-01, 3.19561e-01, 2.15144e-01, 1.47464e-01, 1.01087e-01, 6.93181e-02, 4.75044e-02}, true) and
		edm::ParameterDescription<std::vector<double>>("EEPulseShapeTemplate", {1.16442e-01, 7.56246e-01, 1.00000e+00, 8.97182e-01, 6.86831e-01, 4.91506e-01, 3.44111e-01, 2.45731e-01, 1.74115e-01, 1.23361e-01, 8.74288e-02, 6.19570e-02}, true)));
  
  psd0.addNode((edm::ParameterDescription<std::string>("EEdigiCollection", "", true) and
		edm::ParameterDescription<std::string>("EBdigiCollection", "", true) and
		edm::ParameterDescription<std::string>("ESdigiCollection", "", true) and
		edm::ParameterDescription<bool>("UseLCcorrection", true, false) and
		edm::ParameterDescription<std::vector<double>>("EBCorrNoiseMatrixG12", {1.00000, 0.71073, 0.55721, 0.46089, 0.40449, 0.35931, 0.33924, 0.32439, 0.31581, 0.30481 }, true) and
		edm::ParameterDescription<std::vector<double>>("EECorrNoiseMatrixG12", {1.00000, 0.71373, 0.44825, 0.30152, 0.21609, 0.14786, 0.11772, 0.10165, 0.09465, 0.08098 }, true) and
		edm::ParameterDescription<std::vector<double>>("EBCorrNoiseMatrixG06", {1.00000, 0.70946, 0.58021, 0.49846, 0.45006, 0.41366, 0.39699, 0.38478, 0.37847, 0.37055 }, true) and
		edm::ParameterDescription<std::vector<double>>("EECorrNoiseMatrixG06", {1.00000, 0.71217, 0.47464, 0.34056, 0.26282, 0.20287, 0.17734, 0.16256, 0.15618, 0.14443 }, true) and
		edm::ParameterDescription<std::vector<double>>("EBCorrNoiseMatrixG01", {1.00000, 0.73354, 0.64442, 0.58851, 0.55425, 0.53082, 0.51916, 0.51097, 0.50732, 0.50409 }, true) and
		edm::ParameterDescription<std::vector<double>>("EECorrNoiseMatrixG01", {1.00000, 0.72698, 0.62048, 0.55691, 0.51848, 0.49147, 0.47813, 0.47007, 0.46621, 0.46265 }, true) and
		edm::ParameterDescription<bool>("EcalPreMixStage1", false, true) and
		edm::ParameterDescription<bool>("EcalPreMixStage2", false, true)));

  psd0.addOptionalNode((edm::ParameterDescription<std::vector<double>>("EBPulseShapeCovariance", {3.001e-06,  1.233e-05,  0.000e+00, -4.416e-06, -4.571e-06, -3.614e-06, -2.636e-06, -1.286e-06, -8.410e-07, -5.296e-07,  0.000e+00,  0.000e+00, 
 	   1.233e-05,  6.154e-05,  0.000e+00, -2.200e-05, -2.309e-05, -1.838e-05, -1.373e-05, -7.334e-06, -5.088e-06, -3.745e-06, -2.428e-06,  0.000e+00, 
 	   0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00, 
 	   -4.416e-06, -2.200e-05,  0.000e+00,  8.319e-06,  8.545e-06,  6.792e-06,  5.059e-06,  2.678e-06,  1.816e-06,  1.223e-06,  8.245e-07,  5.589e-07, 
 	   -4.571e-06, -2.309e-05,  0.000e+00,  8.545e-06,  9.182e-06,  7.219e-06,  5.388e-06,  2.853e-06,  1.944e-06,  1.324e-06,  9.083e-07,  6.335e-07, 
 	   -3.614e-06, -1.838e-05,  0.000e+00,  6.792e-06,  7.219e-06,  6.016e-06,  4.437e-06,  2.385e-06,  1.636e-06,  1.118e-06,  7.754e-07,  5.556e-07, 
 	   -2.636e-06, -1.373e-05,  0.000e+00,  5.059e-06,  5.388e-06,  4.437e-06,  3.602e-06,  1.917e-06,  1.322e-06,  9.079e-07,  6.529e-07,  4.752e-07, 
 	   -1.286e-06, -7.334e-06,  0.000e+00,  2.678e-06,  2.853e-06,  2.385e-06,  1.917e-06,  1.375e-06,  9.100e-07,  6.455e-07,  4.693e-07,  3.657e-07, 
 	   -8.410e-07, -5.088e-06,  0.000e+00,  1.816e-06,  1.944e-06,  1.636e-06,  1.322e-06,  9.100e-07,  9.115e-07,  6.062e-07,  4.436e-07,  3.422e-07, 
 	   -5.296e-07, -3.745e-06,  0.000e+00,  1.223e-06,  1.324e-06,  1.118e-06,  9.079e-07,  6.455e-07,  6.062e-07,  7.217e-07,  4.862e-07,  3.768e-07, 
 	   0.000e+00, -2.428e-06,  0.000e+00,  8.245e-07,  9.083e-07,  7.754e-07,  6.529e-07,  4.693e-07,  4.436e-07,  4.862e-07,  6.509e-07,  4.418e-07, 
 	   0.000e+00,  0.000e+00,  0.000e+00,  5.589e-07,  6.335e-07,  5.556e-07,  4.752e-07,  3.657e-07,  3.422e-07,  3.768e-07,  4.418e-07,  6.142e-07}, true) and
     edm::ParameterDescription<std::vector<double>>("EEPulseShapeCovariance", {3.941e-05,  3.333e-05,  0.000e+00, -1.449e-05, -1.661e-05, -1.424e-05, -1.183e-05, -6.842e-06, -4.915e-06, -3.411e-06,  0.000e+00,  0.000e+00, 
 	   3.333e-05,  2.862e-05,  0.000e+00, -1.244e-05, -1.431e-05, -1.233e-05, -1.032e-05, -5.883e-06, -4.154e-06, -2.902e-06, -2.128e-06,  0.000e+00, 
 	   0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00,  0.000e+00, 
 	   -1.449e-05, -1.244e-05,  0.000e+00,  5.840e-06,  6.649e-06,  5.720e-06,  4.812e-06,  2.708e-06,  1.869e-06,  1.330e-06,  9.186e-07,  6.446e-07, 
 	   -1.661e-05, -1.431e-05,  0.000e+00,  6.649e-06,  7.966e-06,  6.898e-06,  5.794e-06,  3.157e-06,  2.184e-06,  1.567e-06,  1.084e-06,  7.575e-07, 
 	   -1.424e-05, -1.233e-05,  0.000e+00,  5.720e-06,  6.898e-06,  6.341e-06,  5.347e-06,  2.859e-06,  1.991e-06,  1.431e-06,  9.839e-07,  6.886e-07, 
 	   -1.183e-05, -1.032e-05,  0.000e+00,  4.812e-06,  5.794e-06,  5.347e-06,  4.854e-06,  2.628e-06,  1.809e-06,  1.289e-06,  9.020e-07,  6.146e-07, 
 	   -6.842e-06, -5.883e-06,  0.000e+00,  2.708e-06,  3.157e-06,  2.859e-06,  2.628e-06,  1.863e-06,  1.296e-06,  8.882e-07,  6.108e-07,  4.283e-07, 
 	   -4.915e-06, -4.154e-06,  0.000e+00,  1.869e-06,  2.184e-06,  1.991e-06,  1.809e-06,  1.296e-06,  1.217e-06,  8.669e-07,  5.751e-07,  3.882e-07, 
 	   -3.411e-06, -2.902e-06,  0.000e+00,  1.330e-06,  1.567e-06,  1.431e-06,  1.289e-06,  8.882e-07,  8.669e-07,  9.522e-07,  6.717e-07,  4.293e-07, 
 	   0.000e+00, -2.128e-06,  0.000e+00,  9.186e-07,  1.084e-06,  9.839e-07,  9.020e-07,  6.108e-07,  5.751e-07,  6.717e-07,  7.911e-07,  5.493e-07, 
 	   0.000e+00,  0.000e+00,  0.000e+00,  6.446e-07,  7.575e-07,  6.886e-07,  6.146e-07,  4.283e-07,  3.882e-07,  4.293e-07,  5.493e-07,  7.027e-07}, true)), true);

 edm::ParameterSetDescription psd;
 psd.addNode(edm::ParameterDescription<std::vector<int>>("activeBXs", {-5,-4,-3,-2,-1,0,1,2,3,4}, true) and
	      edm::ParameterDescription<bool>("ampErrorCalculation", true, true) and
	      edm::ParameterDescription<bool>("useLumiInfoRunHeader", true, true) and
	      edm::ParameterDescription<int>("bunchSpacing", 0, true) and
	      edm::ParameterDescription<bool>("doPrefitEB", false, true) and
	      edm::ParameterDescription<bool>("doPrefitEE", false, true) and
	      edm::ParameterDescription<double>("prefitMaxChiSqEB", 25., true) and
	      edm::ParameterDescription<double>("prefitMaxChiSqEE", 10., true) and
	      edm::ParameterDescription<std::string>("timealgo", "RatioMethod", true) and
	      edm::ParameterDescription<std::vector<double>>("EBtimeFitParameters", {-2.015452e+00, 3.130702e+00, -1.234730e+01, 4.188921e+01, -8.283944e+01, 9.101147e+01, -5.035761e+01, 1.105621e+01}, true) and
	      edm::ParameterDescription<std::vector<double>>("EEtimeFitParameters", {-2.390548e+00, 3.553628e+00, -1.762341e+01, 6.767538e+01, -1.332130e+02, 1.407432e+02, -7.541106e+01, 1.620277e+01}, true) and
	      edm::ParameterDescription<std::vector<double>>("EBamplitudeFitParameters", {1.138,1.652}, true) and
	      edm::ParameterDescription<std::vector<double>>("EEamplitudeFitParameters", {1.890,1.400}, true) and
	      edm::ParameterDescription<double>("EBtimeFitLimits_Lower", 0.2, true) and
	      edm::ParameterDescription<double>("EBtimeFitLimits_Upper", 1.4, true) and
	      edm::ParameterDescription<double>("EEtimeFitLimits_Lower", 0.2, true) and
	      edm::ParameterDescription<double>("EEtimeFitLimits_Upper", 1.4, true) and
	      edm::ParameterDescription<double>("EBtimeConstantTerm", .6, true) and
	      edm::ParameterDescription<double>("EEtimeConstantTerm", 1.0, true) and
	      edm::ParameterDescription<double>("EBtimeNconst", 28.5, true) and
	      edm::ParameterDescription<double>("EEtimeNconst", 31.8, true) and
	      edm::ParameterDescription<double>("outOfTimeThresholdGain12pEB", 5, true) and
	      edm::ParameterDescription<double>("outOfTimeThresholdGain12mEB", 5, true) and
	      edm::ParameterDescription<double>("outOfTimeThresholdGain61pEB", 5, true) and
	      edm::ParameterDescription<double>("outOfTimeThresholdGain61mEB", 5, true) and
	      edm::ParameterDescription<double>("outOfTimeThresholdGain12pEE", 1000, true) and
	      edm::ParameterDescription<double>("outOfTimeThresholdGain12mEE", 1000, true) and
	      edm::ParameterDescription<double>("outOfTimeThresholdGain61pEE", 1000, true) and
	      edm::ParameterDescription<double>("outOfTimeThresholdGain61mEE", 1000, true) and
	      edm::ParameterDescription<double>("amplitudeThresholdEB", 10, true) and
	      edm::ParameterDescription<double>("amplitudeThresholdEE", 10, true) and
	      edm::ParameterDescription<double>("ebSpikeThreshold", 1.042, true) and
	      edm::ParameterDescription<std::vector<double>>("ebPulseShape", {5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194}, true) and
	      edm::ParameterDescription<std::vector<double>>("eePulseShape", {5.2e-05,-5.26e-05 , 6.66e-05, 0.1168, 0.7575, 1.,  0.8876, 0.6732, 0.4741,  0.3194}, true) and
	      edm::ParameterDescription<bool>("kPoorRecoFlagEB", true, true) and
	      edm::ParameterDescription<bool>("kPoorRecoFlagEE", false, true) and
	      edm::ParameterDescription<double>("chi2ThreshEB_", 65.0, true) and
	      edm::ParameterDescription<double>("chi2ThreshEE_", 50.0, true) and
	      edm::ParameterDescription<edm::ParameterSetDescription>("EcalPulseShapeParameters", psd0, true));

 return psd;
}



#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerMultiFit, "EcalUncalibRecHitWorkerMultiFit" );
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitFillDescriptionWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitFillDescriptionWorkerFactory, EcalUncalibRecHitWorkerMultiFit, "EcalUncalibRecHitWorkerMultiFit" );
