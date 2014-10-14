#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerMultiFit.h"

#include "FWCore/Framework/interface/EventSetup.h"
#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "CondFormats/DataRecord/interface/EcalGainRatiosRcd.h"
#include "CondFormats/DataRecord/interface/EcalPedestalsRcd.h"
#include "CondFormats/DataRecord/interface/EcalWeightXtalGroupsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTBWeightsRcd.h"
#include "CondFormats/DataRecord/interface/EcalSampleMaskRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeCalibConstantsRcd.h"
#include "CondFormats/DataRecord/interface/EcalTimeOffsetConstantRcd.h"


EcalUncalibRecHitWorkerMultiFit::EcalUncalibRecHitWorkerMultiFit(const edm::ParameterSet&ps) :
  EcalUncalibRecHitWorkerBaseClass(ps),
  noisecorEBg12(SampleMatrix::Zero()), noisecorEEg12(SampleMatrix::Zero()), noisecorEKg12(SampleMatrix::Zero()),
  noisecorEBg6(SampleMatrix::Zero()), noisecorEEg6(SampleMatrix::Zero()), noisecorEKg6(SampleMatrix::Zero()),
  noisecorEBg1(SampleMatrix::Zero()), noisecorEEg1(SampleMatrix::Zero()), noisecorEKg1(SampleMatrix::Zero()),
  fullpulseEB(FullSampleVector::Zero()),fullpulseEE(FullSampleVector::Zero()),fullpulseEK(FullSampleVector::Zero()),
  fullpulsecovEB(FullSampleMatrix::Zero()),fullpulsecovEE(FullSampleMatrix::Zero()),fullpulsecovEK(FullSampleMatrix::Zero()) {

  // get the pulse shape, amplitude covariances and noise correlations
  EcalPulseShapeParameters_ = ps.getParameter<edm::ParameterSet>("EcalPulseShapeParameters");
  fillInputs(EcalPulseShapeParameters_);

  // get the BX for the pulses to be activated
  std::vector<int32_t> activeBXs = ps.getParameter< std::vector<int32_t> >("activeBXs");
  activeBX.resize(activeBXs.size());
  for (unsigned int ibx=0; ibx<activeBXs.size(); ++ibx) {
    activeBX.coeffRef(ibx) = activeBXs[ibx];
  }
  
  noiseMatrixAsCovarianceEB_ = ps.getParameter<bool>("noiseMatrixAsCovarianceEB");
  noiseMatrixAsCovarianceEE_ = ps.getParameter<bool>("noiseMatrixAsCovarianceEE");
  noiseMatrixAsCovarianceEK_ = ps.getParameter<bool>("noiseMatrixAsCovarianceEK");

  // uncertainty calculation (CPU intensive)
  ampErrorCalculation_ = ps.getParameter<bool>("ampErrorCalculation");

  // algorithm to be used for timing
  timealgo_ = ps.getParameter<std::string>("timealgo");
  
  // ratio method parameters
  EBtimeFitParameters_ = ps.getParameter<std::vector<double> >("EBtimeFitParameters"); 
  EEtimeFitParameters_ = ps.getParameter<std::vector<double> >("EEtimeFitParameters"); 
  EKtimeFitParameters_ = ps.getParameter<std::vector<double> >("EKtimeFitParameters"); 
  EBamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EBamplitudeFitParameters");
  EEamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EEamplitudeFitParameters");
  EKamplitudeFitParameters_ = ps.getParameter<std::vector<double> >("EKamplitudeFitParameters");
  EBtimeFitLimits_.first  = ps.getParameter<double>("EBtimeFitLimits_Lower");
  EBtimeFitLimits_.second = ps.getParameter<double>("EBtimeFitLimits_Upper");
  EEtimeFitLimits_.first  = ps.getParameter<double>("EEtimeFitLimits_Lower");
  EEtimeFitLimits_.second = ps.getParameter<double>("EEtimeFitLimits_Upper");
  EKtimeFitLimits_.first  = ps.getParameter<double>("EKtimeFitLimits_Lower");
  EKtimeFitLimits_.second = ps.getParameter<double>("EKtimeFitLimits_Upper");  
  EBtimeConstantTerm_=ps.getParameter<double>("EBtimeConstantTerm");
  EEtimeConstantTerm_=ps.getParameter<double>("EEtimeConstantTerm");
  EKtimeConstantTerm_=ps.getParameter<double>("EKtimeConstantTerm");

  // amplitude-dependent correction of time
  doEBtimeCorrection_      = ps.getParameter<bool>("doEBtimeCorrection");
  doEEtimeCorrection_      = ps.getParameter<bool>("doEEtimeCorrection");
  doEKtimeCorrection_      = ps.getParameter<bool>("doEKtimeCorrection");
  EBtimeCorrAmplitudeBins_ = ps.getParameter<std::vector<double> >("EBtimeCorrAmplitudeBins"); 
  EBtimeCorrShiftBins_     = ps.getParameter<std::vector<double> >("EBtimeCorrShiftBins"); 
  EEtimeCorrAmplitudeBins_ = ps.getParameter<std::vector<double> >("EEtimeCorrAmplitudeBins"); 
  EEtimeCorrShiftBins_     = ps.getParameter<std::vector<double> >("EEtimeCorrShiftBins"); 
  EKtimeCorrAmplitudeBins_ = ps.getParameter<std::vector<double> >("EKtimeCorrAmplitudeBins"); 
  EKtimeCorrShiftBins_     = ps.getParameter<std::vector<double> >("EKtimeCorrShiftBins"); 
  if(EBtimeCorrAmplitudeBins_.size() != EBtimeCorrShiftBins_.size()) {
    doEBtimeCorrection_ = false;
    edm::LogError("EcalRecHitError") << "Size of EBtimeCorrAmplitudeBins different from EBtimeCorrShiftBins. Forcing no time corrections for EB. ";
  }
  if(EEtimeCorrAmplitudeBins_.size() != EEtimeCorrShiftBins_.size()) {
    doEEtimeCorrection_ = false;
    edm::LogError("EcalRecHitError") << "Size of EEtimeCorrAmplitudeBins different from EEtimeCorrShiftBins. Forcing no time corrections for EE. ";
  }
  if(EKtimeCorrAmplitudeBins_.size() != EKtimeCorrShiftBins_.size()) {
    doEKtimeCorrection_ = false;
    edm::LogError("EcalRecHitError") << "Size of EKtimeCorrAmplitudeBins different from EKtimeCorrShiftBins. Forcing no time corrections for EK. ";
  }  
  
  // leading edge parameters
  ebPulseShape_ = ps.getParameter<std::vector<double> >("ebPulseShape");
  eePulseShape_ = ps.getParameter<std::vector<double> >("eePulseShape");
  ekPulseShape_ = ps.getParameter<std::vector<double> >("ekPulseShape");

  // chi2 parameters for flags determination
  kPoorRecoFlagEB_ = ps.getParameter<bool>("kPoorRecoFlagEB");
  kPoorRecoFlagEE_ = ps.getParameter<bool>("kPoorRecoFlagEE");;
  kPoorRecoFlagEK_ = ps.getParameter<bool>("kPoorRecoFlagEK");;
  chi2ThreshEB_=ps.getParameter<double>("chi2ThreshEB_");
  chi2ThreshEE_=ps.getParameter<double>("chi2ThreshEE_");
  chi2ThreshEK_=ps.getParameter<double>("chi2ThreshEK_");

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

}

double EcalUncalibRecHitWorkerMultiFit::timeCorrectionEB(float ampliEB){
  // computed initially in ns. Than turned in the BX's, as EcalUncalibratedRecHit need be.
  double theCorrection=0;

  
  int myBin = -1;
  for (int bin=0; bin<(int)EBtimeCorrAmplitudeBins_.size(); bin++ ){
    if(ampliEB > EBtimeCorrAmplitudeBins_.at(bin)) {
      myBin = bin;     }
    else break;
  }
  
  if (myBin == -1)
    {
      theCorrection = EBtimeCorrShiftBins_.at(0);
    }    
  else if  ( myBin == ((int)(EBtimeCorrAmplitudeBins_.size()-1))   ) 
    {
      theCorrection = EBtimeCorrShiftBins_.at( myBin );      
    }    
  else if  ( -1 < myBin   &&   myBin <  ((int)EBtimeCorrAmplitudeBins_.size()-1) )
    {
      // interpolate linearly between two assingned points
      theCorrection  = ( EBtimeCorrShiftBins_.at(myBin+1) - EBtimeCorrShiftBins_.at(myBin) );
      theCorrection *= ( ((double)ampliEB) -  EBtimeCorrAmplitudeBins_.at(myBin) ) / ( EBtimeCorrAmplitudeBins_.at(myBin+1) - EBtimeCorrAmplitudeBins_.at(myBin) );
      theCorrection += EBtimeCorrShiftBins_.at(myBin);
    }
  else
    {
      edm::LogError("EcalRecHitError") << "Assigning time correction impossible. Setting it to 0 ";
      theCorrection = 0.;
    }

  // convert ns into clocks
  return theCorrection/25.;
}


double EcalUncalibRecHitWorkerMultiFit::timeCorrectionEE(float ampliEE){
  // computed initially in ns. Than turned in the BX's, as EcalUncalibratedRecHit need be.
  double theCorrection=0;
  
  int myBin = -1;
  for (int bin=0; bin<(int)EEtimeCorrAmplitudeBins_.size(); bin++ ){
    if(ampliEE > EEtimeCorrAmplitudeBins_.at(bin)) {
      myBin = bin;     }
    else break;
  }
  
  if (myBin == -1)
    {
      theCorrection = EEtimeCorrShiftBins_.at(0);
    }    
  else if  ( myBin == ((int)(EEtimeCorrAmplitudeBins_.size()-1))   ) 
    {
      theCorrection = EEtimeCorrShiftBins_.at( myBin );      
    }    
  else if  ( -1 < myBin   &&   myBin <  ((int)EEtimeCorrAmplitudeBins_.size()-1) )
    {
      // interpolate linearly between two assingned points
      theCorrection  = ( EEtimeCorrShiftBins_.at(myBin+1) - EEtimeCorrShiftBins_.at(myBin) );
      theCorrection *= ( ((double)ampliEE) -  EEtimeCorrAmplitudeBins_.at(myBin) ) / ( EEtimeCorrAmplitudeBins_.at(myBin+1) - EEtimeCorrAmplitudeBins_.at(myBin) );
      theCorrection += EEtimeCorrShiftBins_.at(myBin);
    }
  else
    {
      edm::LogError("EcalRecHitError") << "Assigning time correction impossible. Setting it to 0 ";
      theCorrection = 0.;
    }
  
  // convert ns into clocks
  return theCorrection/25.;
}

double EcalUncalibRecHitWorkerMultiFit::timeCorrectionEK(float ampliEK){
  // computed initially in ns. Than turned in the BX's, as EcalUncalibratedRecHit need be.
  double theCorrection=0;
  
  int myBin = -1;
  for (int bin=0; bin<(int)EKtimeCorrAmplitudeBins_.size(); bin++ ){
    if(ampliEK > EKtimeCorrAmplitudeBins_.at(bin)) {
      myBin = bin;     }
    else break;
  }
  
  if (myBin == -1)
    {
      theCorrection = EKtimeCorrShiftBins_.at(0);
    }    
  else if  ( myBin == ((int)(EKtimeCorrAmplitudeBins_.size()-1))   ) 
    {
      theCorrection = EKtimeCorrShiftBins_.at( myBin );      
    }    
  else if  ( -1 < myBin   &&   myBin <  ((int)EKtimeCorrAmplitudeBins_.size()-1) )
    {
      // interpolate linearly between two assingned points
      theCorrection  = ( EKtimeCorrShiftBins_.at(myBin+1) - EKtimeCorrShiftBins_.at(myBin) );
      theCorrection *= ( ((double)ampliEK) -  EKtimeCorrAmplitudeBins_.at(myBin) ) / ( EKtimeCorrAmplitudeBins_.at(myBin+1) - EKtimeCorrAmplitudeBins_.at(myBin) );
      theCorrection += EKtimeCorrShiftBins_.at(myBin);
    }
  else
    {
      edm::LogError("EcalRecHitError") << "Assigning time correction impossible. Setting it to 0 ";
      theCorrection = 0.;
    }
  
  // convert ns into clocks
  return theCorrection/25.;
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
        bool noiseMatrixAsCovariance = false;
        
        if (detid.subdetId()==EcalEndcap) {
          unsigned int hashedIndex = EEDetId(detid).hashedIndex();
          aped  = &peds->endcap(hashedIndex);
          aGain = &gains->endcap(hashedIndex);
          gid   = &grps->endcap(hashedIndex);
          noiseMatrixAsCovariance = noiseMatrixAsCovarianceEE_;
        } else if (detid.subdetId()==EcalShashlik){
          unsigned int hashedIndex = 10;
          aped  = &peds->shashlik(hashedIndex);
          aGain = &gains->shashlik(hashedIndex);
          gid   = &grps->shashlik(hashedIndex);
          noiseMatrixAsCovariance = noiseMatrixAsCovarianceEK_;
        } else {
          unsigned int hashedIndex = EBDetId(detid).hashedIndex();
          aped  = &peds->barrel(hashedIndex);
          aGain = &gains->barrel(hashedIndex);
          gid   = &grps->barrel(hashedIndex);
          noiseMatrixAsCovariance = noiseMatrixAsCovarianceEB_;
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
        //int leadingSample = -1;
        //        if (detid.subdetId()==EcalEndcap) {
        int leadingSample = ((EcalDataFrame)(*itdg)).lastUnsaturatedSample();
                //} else if(detid.subdetId()==EcalShashlik) { 
                //      leadingSample = ((EcalDataFrame)(*itdg)).lastUnsaturatedSample();
        //} else {
        //leadingSample = ((EcalDataFrame)(*itdg)).lastUnsaturatedSample();
        //}

        if ( leadingSample >= 0 ) { // saturation
                if ( leadingSample != 4 ) {
                        // all samples different from the fifth are not reliable for the amplitude estimation
                        // put by default the energy at the saturation threshold and flag as saturated
                        float sratio = 1;
                        if ( detid.subdetId()==EcalBarrel) {
                                sratio = ebPulseShape_[5] / ebPulseShape_[4];
                        } else if( detid.subdetId()==EcalShashlik) {
                          sratio = ekPulseShape_[5] / ekPulseShape_[4];
                        } else {
                                sratio = eePulseShape_[5] / eePulseShape_[4];
                        }
                        uncalibRecHit = EcalUncalibratedRecHit( (*itdg).id(), 4095*12*sratio, 0, 0, 0); //Shervin 4095 hardcoded!!!!
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
                        } else if (detid.subdetId()==EcalShashlik) {
                          leadingEdgeMethod_shashlik_.setPulseShape( ekPulseShape_ );
                          // float mult = (float)eePulseShape_.size() / (float)(*itdg).size();
                          // bin (or some analogous mapping) will be used instead of the leadingSample
                          //int bin  = (int)(( (mult * leadingSample + mult/2) * clockToNsConstant + itimeconst ) / clockToNsConstant);
                          // bin is not uset for the moment
                          leadingEdgeMethod_shashlik_.setLeadingEdgeSample( leadingSample );
                          uncalibRecHit = leadingEdgeMethod_shashlik_.makeRecHit(*itdg, pedVec, gainRatios, 0, 0);
                          uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kLeadingEdgeRecovered );
                          leadingEdgeMethod_shashlik_.setLeadingEdgeSample( -1 );
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
                                
                uncalibRecHit = multiFitMethod_.makeRecHit(*itdg, aped, aGain, noisecormat,fullpulse,fullpulsecov,activeBX,noiseMatrixAsCovariance);
                
                // === time computation ===
                if(timealgo_.compare("RatioMethod")==0) {
                  // ratio method
                  float const clockToNsConstant = 25.;
                  if (detid.subdetId()==EcalEndcap) {
                                  ratioMethod_endcap_.init( *itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios );
                                  ratioMethod_endcap_.computeTime( EEtimeFitParameters_, EEtimeFitLimits_, EEamplitudeFitParameters_ );
                                  ratioMethod_endcap_.computeAmplitude( EEamplitudeFitParameters_);
                                  EcalUncalibRecHitRatioMethodAlgo<EEDataFrame>::CalculatedRecHit crh = ratioMethod_endcap_.getCalculatedRecHit();
                                  double theTimeCorrectionEE=0;
                                  if(doEEtimeCorrection_) theTimeCorrectionEE = timeCorrectionEE( uncalibRecHit.amplitude() );
                                  uncalibRecHit.setJitter( crh.timeMax - 5 + theTimeCorrectionEE);
                                  uncalibRecHit.setJitterError( std::sqrt(pow(crh.timeError,2) + std::pow(EEtimeConstantTerm_,2)/std::pow(clockToNsConstant,2)) );
                  }else                 if (detid.subdetId()==EcalShashlik) {
                                  ratioMethod_shashlik_.init( *itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios );
                                  ratioMethod_shashlik_.computeTime( EKtimeFitParameters_, EKtimeFitLimits_, EKamplitudeFitParameters_ );
                                  ratioMethod_shashlik_.computeAmplitude( EKamplitudeFitParameters_);
                                  EcalUncalibRecHitRatioMethodAlgo<EKDataFrame>::CalculatedRecHit crh = ratioMethod_shashlik_.getCalculatedRecHit();
                                  double theTimeCorrectionEK=0;
                                  if(doEKtimeCorrection_) theTimeCorrectionEK = timeCorrectionEK( uncalibRecHit.amplitude() );
                                  uncalibRecHit.setJitter( crh.timeMax - 5 + theTimeCorrectionEK);
                                  uncalibRecHit.setJitterError( std::sqrt(pow(crh.timeError,2) + std::pow(EKtimeConstantTerm_,2)/std::pow(clockToNsConstant,2)) );
                  } else {
                                  ratioMethod_barrel_.init( *itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios );
                                  ratioMethod_barrel_.fixMGPAslew(*itdg);
                                  ratioMethod_barrel_.computeTime( EBtimeFitParameters_, EBtimeFitLimits_, EBamplitudeFitParameters_ );
                                  ratioMethod_barrel_.computeAmplitude( EBamplitudeFitParameters_);
                                  EcalUncalibRecHitRatioMethodAlgo<EBDataFrame>::CalculatedRecHit crh = ratioMethod_barrel_.getCalculatedRecHit();
                                  double theTimeCorrectionEB=0;
                                  if(doEBtimeCorrection_) theTimeCorrectionEB = timeCorrectionEB( uncalibRecHit.amplitude() );

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
                  } else if (detid.subdetId()==EcalShashlik) {
                    timerh = weightsMethod_shashlik_.time( *itdg, amplitudes, aped, aGain, fullpulse, weights);                
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
  const std::vector<double> ekCorMatG12 = params.getParameter< std::vector<double> >("EKCorrNoiseMatrixG12");
  const std::vector<double> ebCorMatG06 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG06");
  const std::vector<double> eeCorMatG06 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG06");
  const std::vector<double> ekCorMatG06 = params.getParameter< std::vector<double> >("EKCorrNoiseMatrixG06");
  const std::vector<double> ebCorMatG01 = params.getParameter< std::vector<double> >("EBCorrNoiseMatrixG01");
  const std::vector<double> eeCorMatG01 = params.getParameter< std::vector<double> >("EECorrNoiseMatrixG01");
  const std::vector<double> ekCorMatG01 = params.getParameter< std::vector<double> >("EKCorrNoiseMatrixG01");
  
  int nnoise = ebCorMatG12.size();

  // fill correlation matrices: noise (HF (+) LF)
  for (int i=0; i<nnoise; ++i) {
    for (int j=0; j<nnoise; ++j) {
      int vidx = std::abs(j-i);
      noisecorEBg12(i,j) = ebCorMatG12[vidx];
      noisecorEEg12(i,j) = eeCorMatG12[vidx];
      noisecorEKg12(i,j) = ekCorMatG12[vidx];
      noisecorEBg6(i,j)  = ebCorMatG06[vidx];
      noisecorEEg6(i,j)  = eeCorMatG06[vidx];
      noisecorEKg6(i,j)  = ekCorMatG06[vidx];
      noisecorEBg1(i,j)  = ebCorMatG01[vidx];
      noisecorEEg1(i,j)  = eeCorMatG01[vidx];        
      noisecorEKg1(i,j)  = ekCorMatG01[vidx];        
    }
  }
  
  // fill shape: from simulation for samples 3-9, from alpha/beta shape for 10-14
  const std::vector<double> ebPulse = params.getParameter< std::vector<double> >("EBPulseShapeTemplate");
  const std::vector<double> eePulse = params.getParameter< std::vector<double> >("EEPulseShapeTemplate");
  const std::vector<double> ekPulse = params.getParameter< std::vector<double> >("EKPulseShapeTemplate");
  int nShapeSamples = ebPulse.size();
  for (int i=0; i<nShapeSamples; ++i) {
    fullpulseEB(i+7) = ebPulse[i];
    fullpulseEE(i+7) = eePulse[i];
    fullpulseEE(i+7) = ekPulse[i];
  }

  const std::vector<double> ebPulseCov = params.getParameter< std::vector<double> >("EBPulseShapeCovariance");
  const std::vector<double> eePulseCov = params.getParameter< std::vector<double> >("EEPulseShapeCovariance");
  const std::vector<double> ekPulseCov = params.getParameter< std::vector<double> >("EKPulseShapeCovariance");
  for(int k=0; k<std::pow(nShapeSamples,2); ++k) {
    int i = k/nShapeSamples;
    int j = k%nShapeSamples;
    fullpulsecovEB(i+7,j+7) = ebPulseCov[k];
    fullpulsecovEE(i+7,j+7) = eePulseCov[k];
    fullpulsecovEK(i+7,j+7) = ekPulseCov[k];
  }

}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerMultiFit, "EcalUncalibRecHitWorkerMultiFit" );
