#include "RecoLocalCalo/EcalRecProducers/plugins/EcalUncalibRecHitWorkerGlobal.h"

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

EcalUncalibRecHitWorkerGlobal::EcalUncalibRecHitWorkerGlobal(const edm::ParameterSet&ps) :
        EcalUncalibRecHitWorkerBaseClass(ps)
{
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
        EBtimeNconst_=ps.getParameter<double>("EBtimeNconst");
        EEtimeConstantTerm_=ps.getParameter<double>("EEtimeConstantTerm");
        EEtimeNconst_=ps.getParameter<double>("EEtimeNconst");
        EKtimeConstantTerm_=ps.getParameter<double>("EKtimeConstantTerm");
        EKtimeNconst_=ps.getParameter<double>("EKtimeNconst");
        outOfTimeThreshG12pEB_ = ps.getParameter<double>("outOfTimeThresholdGain12pEB");
        outOfTimeThreshG12mEB_ = ps.getParameter<double>("outOfTimeThresholdGain12mEB");
        outOfTimeThreshG61pEB_ = ps.getParameter<double>("outOfTimeThresholdGain61pEB");
        outOfTimeThreshG61mEB_ = ps.getParameter<double>("outOfTimeThresholdGain61mEB");
        outOfTimeThreshG12pEE_ = ps.getParameter<double>("outOfTimeThresholdGain12pEE");
        outOfTimeThreshG12mEE_ = ps.getParameter<double>("outOfTimeThresholdGain12mEE");
        outOfTimeThreshG61pEE_ = ps.getParameter<double>("outOfTimeThresholdGain61pEE");
        outOfTimeThreshG61mEE_ = ps.getParameter<double>("outOfTimeThresholdGain61mEE");
        outOfTimeThreshG12pEK_ = ps.getParameter<double>("outOfTimeThresholdGain12pEK");
        outOfTimeThreshG12mEK_ = ps.getParameter<double>("outOfTimeThresholdGain12mEK");
        outOfTimeThreshG61pEK_ = ps.getParameter<double>("outOfTimeThresholdGain61pEK");
        outOfTimeThreshG61mEK_ = ps.getParameter<double>("outOfTimeThresholdGain61mEK");
        amplitudeThreshEB_ = ps.getParameter<double>("amplitudeThresholdEB");
        amplitudeThreshEE_ = ps.getParameter<double>("amplitudeThresholdEE");
        amplitudeThreshEK_ = ps.getParameter<double>("amplitudeThresholdEK");
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

	// spike threshold
        ebSpikeThresh_ = ps.getParameter<double>("ebSpikeThreshold");
        // leading edge parameters
        ebPulseShape_ = ps.getParameter<std::vector<double> >("ebPulseShape");
        eePulseShape_ = ps.getParameter<std::vector<double> >("eePulseShape");
        ekPulseShape_ = ps.getParameter<std::vector<double> >("ekPulseShape");
	// chi2 parameters
        kPoorRecoFlagEB_ = ps.getParameter<bool>("kPoorRecoFlagEB");
	kPoorRecoFlagEE_ = ps.getParameter<bool>("kPoorRecoFlagEE");;
	kPoorRecoFlagEK_ = ps.getParameter<bool>("kPoorRecoFlagEK");;
        chi2ThreshEB_=ps.getParameter<double>("chi2ThreshEB_");
	chi2ThreshEE_=ps.getParameter<double>("chi2ThreshEE_");
	chi2ThreshEK_=ps.getParameter<double>("chi2ThreshEK_");
        EBchi2Parameters_ = ps.getParameter<std::vector<double> >("EBchi2Parameters");
        EEchi2Parameters_ = ps.getParameter<std::vector<double> >("EEchi2Parameters");
        EKchi2Parameters_ = ps.getParameter<std::vector<double> >("EKchi2Parameters");
}

void
EcalUncalibRecHitWorkerGlobal::set(const edm::EventSetup& es)
{
        // common setup
        es.get<EcalGainRatiosRcd>().get(gains);
        es.get<EcalPedestalsRcd>().get(peds);

        // for the weights method
        es.get<EcalWeightXtalGroupsRcd>().get(grps);
        es.get<EcalTBWeightsRcd>().get(wgts);

	// which of the samples need be used
	es.get<EcalSampleMaskRcd>().get(sampleMaskHand_);

        // for the ratio method

        // for the leading edge method
        es.get<EcalTimeCalibConstantsRcd>().get(itime);
        es.get<EcalTimeOffsetConstantRcd>().get(offtime);
}


// check saturation: 5 samples with gainId = 0
template < class C >
int EcalUncalibRecHitWorkerGlobal::isSaturated(const C & dataFrame)
{
        //bool saturated_ = 0;
        int cnt;
        for (int j = 0; j < C::MAXSAMPLES - 5; ++j) {
                cnt = 0;
                for (int i = j; i < (j + 5) && i < C::MAXSAMPLES; ++i) {
                        if ( dataFrame.sample(i).gainId() == 0 ) ++cnt;
                }
                if ( cnt == 5 ) return j-1 ; // the last unsaturated sample
        }
        return -1; // no saturation found
}


double EcalUncalibRecHitWorkerGlobal::timeCorrectionEB(float ampliEB){
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


double EcalUncalibRecHitWorkerGlobal::timeCorrectionEE(float ampliEE){
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

double EcalUncalibRecHitWorkerGlobal::timeCorrectionEK(float ampliEK){
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
EcalUncalibRecHitWorkerGlobal::run( const edm::Event & evt,
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
	float offsetTime = 0;

        if (detid.subdetId()==EcalEndcap) {
                unsigned int hashedIndex = EEDetId(detid).hashedIndex();
                aped  = &peds->endcap(hashedIndex);
                aGain = &gains->endcap(hashedIndex);
                gid   = &grps->endcap(hashedIndex);
		offsetTime = offtime->getEEValue();
	} else if (detid.subdetId()==EcalShashlik){
	  unsigned int hashedIndex = 10;
	  aped  = &peds->shashlik(hashedIndex);
	  aGain = &gains->shashlik(hashedIndex);
	  gid   = &grps->shashlik(hashedIndex);
	  offsetTime = offtime->getEEValue(); //Shervin
        } else {
                unsigned int hashedIndex = EBDetId(detid).hashedIndex();
                aped  = &peds->barrel(hashedIndex);
                aGain = &gains->barrel(hashedIndex);
                gid   = &grps->barrel(hashedIndex);
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
		uncalibRecHit.setOutOfTimeChi2(0);
        } else {
                // weights method
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

                // get uncalibrated recHit from weights
		if (detid.subdetId()==EcalEndcap) {
	    	     uncalibRecHit = weightsMethod_endcap_.makeRecHit(*itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEEShape);
		}		if (detid.subdetId()==EcalShashlik) {
	    	     uncalibRecHit = weightsMethod_shashlik_.makeRecHit(*itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEKShape);} else {
		     uncalibRecHit = weightsMethod_barrel_.makeRecHit(*itdg, pedVec, pedRMSVec, gainRatios, weights, testbeamEBShape);
		}

                // === time computation ===
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
                                uncalibRecHit.setOutOfTimeEnergy( crh.amplitudeMax );
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
		}else                 if (detid.subdetId()==EcalShashlik) {
    		                ratioMethod_shashlik_.init( *itdg, *sampleMask_, pedVec, pedRMSVec, gainRatios );
                                ratioMethod_shashlik_.computeTime( EKtimeFitParameters_, EKtimeFitLimits_, EKamplitudeFitParameters_ );
                                ratioMethod_shashlik_.computeAmplitude( EKamplitudeFitParameters_);
                                EcalUncalibRecHitRatioMethodAlgo<EKDataFrame>::CalculatedRecHit crh = ratioMethod_shashlik_.getCalculatedRecHit();
				double theTimeCorrectionEK=0;
				if(doEKtimeCorrection_) theTimeCorrectionEK = timeCorrectionEK( uncalibRecHit.amplitude() );
                                uncalibRecHit.setJitter( crh.timeMax - 5 + theTimeCorrectionEK);
                                uncalibRecHit.setJitterError( std::sqrt(pow(crh.timeError,2) + std::pow(EKtimeConstantTerm_,2)/std::pow(clockToNsConstant,2)) );
                                uncalibRecHit.setOutOfTimeEnergy( crh.amplitudeMax );
				// consider flagging as kOutOfTime only if above noise
				if (uncalibRecHit.amplitude() > pedRMSVec[0] * amplitudeThreshEK_){
				  float outOfTimeThreshP = outOfTimeThreshG12pEK_;
				  float outOfTimeThreshM = outOfTimeThreshG12mEK_;
				  // determine if gain has switched away from gainId==1 (x12 gain)
				  // and determine cuts (number of 'sigmas') to ose for kOutOfTime
				  // >3k ADC is necessasry condition for gain switch to occur
				  if (uncalibRecHit.amplitude() > 3000.){
				    for (int iSample = 0; iSample < EKDataFrame::MAXSAMPLES; iSample++) {
				      int GainId = ((EcalDataFrame)(*itdg)).sample(iSample).gainId();
				      if (GainId!=1) {
					outOfTimeThreshP = outOfTimeThreshG61pEK_;
					outOfTimeThreshM = outOfTimeThreshG61mEK_;
					break;
				      }
				    }}
				  float correctedTime = (crh.timeMax-5) * clockToNsConstant + itimeconst + offsetTime;
				  float cterm         = EKtimeConstantTerm_;
				  float sigmaped      = pedRMSVec[0];  // approx for lower gains
				  float nterm         = EKtimeNconst_*sigmaped/uncalibRecHit.amplitude();
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
				double theTimeCorrectionEB=0;
				if(doEBtimeCorrection_) theTimeCorrectionEB = timeCorrectionEB( uncalibRecHit.amplitude() );

				uncalibRecHit.setJitter( crh.timeMax - 5 + theTimeCorrectionEB);

                                uncalibRecHit.setJitterError( std::sqrt(std::pow(crh.timeError,2) + std::pow(EBtimeConstantTerm_,2)/std::pow(clockToNsConstant,2)) );
                                uncalibRecHit.setOutOfTimeEnergy( crh.amplitudeMax );
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
		
		// === chi2express ===
		if (detid.subdetId()==EcalEndcap) {
		      
		    double amplitude = uncalibRecHit.amplitude();
		    double amplitudeOutOfTime = uncalibRecHit.outOfTimeEnergy();
                    double jitter= uncalibRecHit.jitter();


		
		    EcalUncalibRecHitRecChi2Algo<EEDataFrame>chi2expressEE_(
				  					    *itdg, 
				  					    amplitude, 
				  					    (itimeconst + offsetTime), 
				  					    amplitudeOutOfTime, 
				  					    jitter, 
				  					    pedVec, 
				  					    pedRMSVec, 
				  					    gainRatios, 
				  					    testbeamEEShape,
									    EEchi2Parameters_
		    );
		    double chi2 = chi2expressEE_.chi2();
		    uncalibRecHit.setChi2(chi2);
		    double chi2OutOfTime = chi2expressEE_.chi2OutOfTime();
		    uncalibRecHit.setOutOfTimeChi2(chi2OutOfTime);

                    if(kPoorRecoFlagEE_)
		    {
		    
		      if (chi2>chi2ThreshEE_) {

			// first check if all samples are ok, if not don't use chi2 to flag
			bool samplesok = true;
			for (int sample =0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
			  if (!sampleMask_->useSampleEE(sample)) {
			    samplesok = false;
			    break;
			  }
			}
			if (samplesok) uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kPoorReco);
		      }


		    }				
		} else if (detid.subdetId()==EcalShashlik) {
		    double amplitude = uncalibRecHit.amplitude();
		    double amplitudeOutOfTime = uncalibRecHit.outOfTimeEnergy();
                    double jitter= uncalibRecHit.jitter();

		    EcalUncalibRecHitRecChi2Algo<EKDataFrame>chi2expressEK_(
				  					    *itdg, 
				  					    amplitude, 
				  					    (itimeconst + offsetTime), 
				  					    amplitudeOutOfTime, 
				  					    jitter, 
				  					    pedVec, 
				  					    pedRMSVec, 
				  					    gainRatios, 
				  					    testbeamEKShape,
									    EKchi2Parameters_
		    );
		    double chi2 = chi2expressEK_.chi2();
		    uncalibRecHit.setChi2(chi2);
		    double chi2OutOfTime = chi2expressEK_.chi2OutOfTime();
		    uncalibRecHit.setOutOfTimeChi2(chi2OutOfTime);

                    if(kPoorRecoFlagEK_)
		    {
		    
		      if (chi2>chi2ThreshEK_) {

			// first check if all samples are ok, if not don't use chi2 to flag
			bool samplesok = true;
			for (int sample =0; sample < EcalDataFrame::MAXSAMPLES; ++sample) {
			  if (!sampleMask_->useSampleEE(sample)) {//Shervin
			    samplesok = false;
			    break;
			  }
			}
			if (samplesok) uncalibRecHit.setFlagBit(EcalUncalibratedRecHit::kPoorReco);
		      }


		    }				
		} else {
		    double amplitude = uncalibRecHit.amplitude();
		    double amplitudeOutOfTime = uncalibRecHit.outOfTimeEnergy();
                    double jitter= uncalibRecHit.jitter();
		  
		    EcalUncalibRecHitRecChi2Algo<EBDataFrame>chi2expressEB_(
		  							    *itdg, 
		  							    amplitude, 
		  							    (itimeconst + offsetTime), 
		  							    amplitudeOutOfTime, 
		  							    jitter, 
		  							    pedVec, 
		 							    pedRMSVec, 
		  							    gainRatios, 
		  							    testbeamEBShape,
							                    EBchi2Parameters_		
		    );
		    double chi2 = chi2expressEB_.chi2();
		    uncalibRecHit.setChi2(chi2);
		    double chi2OutOfTime = chi2expressEB_.chi2OutOfTime();
		    uncalibRecHit.setOutOfTimeChi2(chi2OutOfTime);

                    if(kPoorRecoFlagEB_)
		    {

		      if(chi2>chi2ThreshEB_){
		      	// first check if all samples are ok, if not don't use chi2 to flag
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
		}
        }

	// set flags if gain switch has occurred
	if( ((EcalDataFrame)(*itdg)).hasSwitchToGain6()  ) uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kHasSwitchToGain6 );
	if( ((EcalDataFrame)(*itdg)).hasSwitchToGain1()  ) uncalibRecHit.setFlagBit( EcalUncalibratedRecHit::kHasSwitchToGain1 );

        // put the recHit in the collection
//         if (detid.subdetId()==EcalEndcap) {
// 	  result.push_back( uncalibRecHit );
// 	} else if (detid.subdetId()==EcalShashlik) {
// 	  result.push_back( uncalibRecHit );
//         } else {
	result.push_back( uncalibRecHit );
	//        }

        return true;
}

#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalCalo/EcalRecProducers/interface/EcalUncalibRecHitWorkerFactory.h"
DEFINE_EDM_PLUGIN( EcalUncalibRecHitWorkerFactory, EcalUncalibRecHitWorkerGlobal, "EcalUncalibRecHitWorkerGlobal" );
