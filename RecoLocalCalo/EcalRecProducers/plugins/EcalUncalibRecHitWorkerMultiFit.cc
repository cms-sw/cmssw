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


EcalUncalibRecHitWorkerMultiFit::EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&ps,edm::ConsumesCollector& c) :
  EcalUncalibRecHitWorkerBaseClass(ps,c),
  noisecorEBg12(SampleMatrix::Zero()), noisecorEEg12(SampleMatrix::Zero()),
  noisecorEBg6(SampleMatrix::Zero()), noisecorEEg6(SampleMatrix::Zero()),
  noisecorEBg1(SampleMatrix::Zero()), noisecorEEg1(SampleMatrix::Zero()),
  fullpulseEB(FullSampleVector::Zero()),fullpulseEE(FullSampleVector::Zero()),
  fullpulsecovEB(FullSampleMatrix::Zero()),fullpulsecovEE(FullSampleMatrix::Zero()) {

  // get the pulse shape, amplitude covariances and noise correlations
  EcalPulseShapeParameters_ = ps.getParameter<edm::ParameterSet>("EcalPulseShapeParameters");
  fillInputs(EcalPulseShapeParameters_);

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

  // leading edge parameters
  ebPulseShape_ = ps.getParameter<std::vector<double> >("ebPulseShape");
  eePulseShape_ = ps.getParameter<std::vector<double> >("eePulseShape");

  // chi2 parameters for flags determination
  kPoorRecoFlagEB_ = ps.getParameter<bool>("kPoorRecoFlagEB");
  kPoorRecoFlagEE_ = ps.getParameter<bool>("kPoorRecoFlagEE");;
  chi2ThreshEB_=ps.getParameter<double>("chi2ThreshEB_");
  chi2ThreshEE_=ps.getParameter<double>("chi2ThreshEE_");

}



// EcalUncalibRecHitWorkerMultiFit::EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&ps) :
//   EcalUncalibRecHitWorkerBaseClass(ps)
// {
//         // ratio method parameters
//         EBtimeFitParameters_ = ps.getParameter<std::vector<double> >("EBtimeFitParameters"); 
//         EEtimeFitParameters_ = ps.getParameter<std::vector<double> >("EEtimeFitParameters"); 
//         EBamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EBamplitudeFitParameters");
//         EEamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EEamplitudeFitParameters");
//         EBtimeFitLimits_.first  = ps.getParameter<double>("EBtimeFitLimits_Lower");
//         EBtimeFitLimits_.second = ps.getParameter<double>("EBtimeFitLimits_Upper");
//         EEtimeFitLimits_.first  = ps.getParameter<double>("EEtimeFitLimits_Lower");
//         EEtimeFitLimits_.second = ps.getParameter<double>("EEtimeFitLimits_Upper");
//         EBtimeConstantTerm_=ps.getParameter<double>("EBtimeConstantTerm");
//         EEtimeConstantTerm_=ps.getParameter<double>("EEtimeConstantTerm");
// 
//         // leading edge parameters
//         ebPulseShape_ = ps.getParameter<std::vector<double> >("ebPulseShape");
//         eePulseShape_ = ps.getParameter<std::vector<double> >("eePulseShape");
// 
// }













void
EcalUncalibRecHitWorkerMultiFit::set(const edm::EventSetup& es)
{

        // common setup
        es.get<EcalGainRatiosRcd>().get(gains);
        es.get<EcalPedestalsRcd>().get(peds);

        // for the multifit method
        if(!ampErrorCalculation_) multiFitMethod_.disableErrorCalculation();

        // weights parameters for the time
        es.get<EcalWeightXtalGroupsRcd>().get(grps);
        es.get<EcalTBWeightsRcd>().get(wgts);

	// which of the samples need be used
	es.get<EcalSampleMaskRcd>().get(sampleMaskHand_);

        // for the ratio method

        // for the leading edge method
        es.get<EcalTimeCalibConstantsRcd>().get(itime);
        es.get<EcalTimeOffsetConstantRcd>().get(offtime);

        // for the time correction methods
        es.get<EcalTimeBiasCorrectionsRcd>().get(timeCorrBias_);
}

void
EcalUncalibRecHitWorkerMultiFit::set(const edm::Event& evt)
{

  if (useLumiInfoRunHeader_) {

    int bunchspacing = 450;
    
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
      else {
        bunchspacing = 50;
      }
    }
    else {
      edm::Handle<int> bunchSpacingH;
      evt.getByToken(bunchSpacing_,bunchSpacingH);
      bunchspacing = *bunchSpacingH;
    }
    
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
        
        
        const EcalPedestals::Item * aped = 0;
        const EcalMGPAGainRatio * aGain = 0;
        const EcalXtalGroupId * gid = 0;

        if (detid.subdetId()==EcalEndcap) {
                unsigned int hashedIndex = EEDetId(detid).hashedIndex();
                aped  = &peds->endcap(hashedIndex);
                aGain = &gains->endcap(hashedIndex);
                gid   = &grps->endcap(hashedIndex);
                multiFitMethod_.setDoPrefit(doPrefitEE_);
		multiFitMethod_.setPrefitMaxChiSq(prefitMaxChiSqEE_);
        } else {
                unsigned int hashedIndex = EBDetId(detid).hashedIndex();
                aped  = &peds->barrel(hashedIndex);
                aGain = &gains->barrel(hashedIndex);
                gid   = &grps->barrel(hashedIndex);
                multiFitMethod_.setDoPrefit(doPrefitEB_);
		multiFitMethod_.setPrefitMaxChiSq(prefitMaxChiSqEB_);
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

        
        // === amplitude computation ===
        int leadingSample = ((EcalDataFrame)(*itdg)).lastUnsaturatedSample();

        if ( leadingSample >= 0 ) { // saturation
                if ( leadingSample != 4 ) {
                        // all samples different from the fifth are not reliable for the amplitude estimation
                        // put by default the energy at the saturation threshold and flag as saturated
                        float sratio = 1;
                        if ( detid.subdetId()==EcalBarrel) {
                                sratio = ebPulseShape_[5] / ebPulseShape_[4];
                        } else {
                                sratio = eePulseShape_[5] / eePulseShape_[4];
                        }
			uncalibRecHit = EcalUncalibratedRecHit( (*itdg).id(), 4095*12*sratio, 0, 0, 0);
                        uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kSaturated );
                } else {
                        // float clockToNsConstant = 25.;
                        // reconstruct the rechit
                        if (detid.subdetId()==EcalEndcap) {
                                leadingEdgeMethod_endcap_.setPulseShape( eePulseShape_ );
                                // float mult = (float)eePulseShape_.size() / (float)(*itdg).size();
                                // bin (or some analogous mapping) will be used instead of the leadingSample
                                //int bin  = (int)(( (mult * leadingSample + mult/2) * clockToNsConstant + itimeconst ) / clockToNsConstant);
                                // bin is not uset for the moment
                                leadingEdgeMethod_endcap_.setLeadingEdgeSample( leadingSample );
                                uncalibRecHit = leadingEdgeMethod_endcap_.makeRecHit(*itdg, pedVec, gainRatios, 0, 0);
                                uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kLeadingEdgeRecovered );
                                leadingEdgeMethod_endcap_.setLeadingEdgeSample( -1 );
                        } else {
                                leadingEdgeMethod_barrel_.setPulseShape( ebPulseShape_ );
                                // float mult = (float)ebPulseShape_.size() / (float)(*itdg).size();
                                // bin (or some analogous mapping) will be used instead of the leadingSample
                                //int bin  = (int)(( (mult * leadingSample + mult/2) * clockToNsConstant + itimeconst ) / clockToNsConstant);
                                // bin is not uset for the moment
                                leadingEdgeMethod_barrel_.setLeadingEdgeSample( leadingSample );
                                uncalibRecHit = leadingEdgeMethod_barrel_.makeRecHit(*itdg, pedVec, gainRatios, 0, 0);
                                uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kLeadingEdgeRecovered );
                                leadingEdgeMethod_barrel_.setLeadingEdgeSample( -1 );
                        }
                }
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

void EcalUncalibRecHitWorkerMultiFit::fillInputs(const edm::ParameterSet& params) {

  const std::vector<double> ebCorMatG12 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG12");
  const std::vector<double> eeCorMatG12 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG12");
  const std::vector<double> ebCorMatG06 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG06");
  const std::vector<double> eeCorMatG06 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG06");
  const std::vector<double> ebCorMatG01 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG01");
  const std::vector<double> eeCorMatG01 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG01");
  
  int nnoise = ebCorMatG12.size();

  // fill correlation matrices: noise (HF (+) LF)
  for (int i=0; i<nnoise; ++i) {
    for (int j=0; j<nnoise; ++j) {
      int vidx = std::abs(j-i);
      noisecorEBg12(i,j) = ebCorMatG12[vidx];
      noisecorEEg12(i,j) = eeCorMatG12[vidx];
      noisecorEBg6(i,j)  = ebCorMatG06[vidx];
      noisecorEEg6(i,j)  = eeCorMatG06[vidx];
      noisecorEBg1(i,j)  = ebCorMatG01[vidx];
      noisecorEEg1(i,j)  = eeCorMatG01[vidx];        
    }
  }
  
  // fill shape: from simulation for samples 3-9, from alpha/beta shape for 10-14
  const std::vector<double> ebPulse = params.getParameter< std::vector<double> >("EBPulseShapeTemplate");
  const std::vector<double> eePulse = params.getParameter< std::vector<double> >("EEPulseShapeTemplate");
  int nShapeSamples = ebPulse.size();
  for (int i=0; i<nShapeSamples; ++i) {
    fullpulseEB(i+7) = ebPulse[i];
    fullpulseEE(i+7) = eePulse[i];
  }

  const std::vector<double> ebPulseCov = params.getParameter< std::vector<double> >("EBPulseShapeCovariance");
  const std::vector<double> eePulseCov = params.getParameter< std::vector<double> >("EEPulseShapeCovariance");
  for(int k=0; k<std::pow(nShapeSamples,2); ++k) {
    int i = k/nShapeSamples;
    int j = k%nShapeSamples;
    fullpulsecovEB(i+7,j+7) = ebPulseCov[k];
    fullpulsecovEE(i+7,j+7) = eePulseCov[k];
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerMultiFit, "EcalUncalibRecHitWorkerMultiFit" );
