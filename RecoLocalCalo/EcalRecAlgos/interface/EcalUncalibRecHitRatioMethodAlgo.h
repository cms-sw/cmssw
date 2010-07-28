#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRatioMethodAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRatioMethodAlgo_HH

/** \class EcalUncalibRecHitRatioMethodAlgo
 *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
 *  using a ratio method
 *
 *  $Id: EcalUncalibRecHitRatioMethodAlgo.h,v 1.18 2010/07/28 07:56:43 innocent Exp $
 *  $Date: 2010/07/28 07:56:43 $
 *  $Revision: 1.18 $
 *  \author A. Ledovskoy (Design) - M. Balazs (Implementation)
 */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include <vector>

template < class C, typename Scalar=double > class EcalUncalibRecHitRatioMethodAlgo {
public:
  typedef double InputScalar;
  struct Ratio {
    int index;
    int step;
    Scalar value;
    Scalar error;
  };
  struct Tmax {
    int index;
    int step;
    Scalar value;
    Scalar error;
    Scalar amplitude;
    Scalar chi2;
  };
  struct CalculatedRecHit {
    Scalar amplitudeMax;
    Scalar timeMax;
    Scalar timeError;
    Scalar chi2;
  };

  ~EcalUncalibRecHitRatioMethodAlgo <() { }
  EcalUncalibratedRecHit makeRecHit(const C & dataFrame,
				    const InputScalar *pedestals,
				    const InputScalar* pedestalRMSes,
				    const InputScalar *gainRatios,
				    std::vector < InputScalar >&timeFitParameters,
				    std::vector < InputScalar >&amplitudeFitParameters,
				    std::pair < InputScalar, InputScalar >&timeFitLimits);
  
  // more function to be able to compute
  // amplitude and time separately
  void init( const C &dataFrame, const InputScalar * pedestals, const InputScalar * pedestalRMSes, const InputScalar * gainRatios );

  void computeTime(std::vector < InputScalar >&timeFitParameters, std::pair < InputScalar, InputScalar >&timeFitLimits, std::vector< InputScalar > &amplitudeFitParameters);

  void computeAmplitude( std::vector< InputScalar > &amplitudeFitParameters ) ;

  CalculatedRecHit const & getCalculatedRecHit() const { return calculatedRechit_; };
  
protected:


  void computeAmpChi2(Scalar sumAA, Scalar t, Scalar alpha, Scalar overab, Scalar & chi2, Scalar & amp) const;

  static const size_t amplitudesSize = C::MAXSAMPLES;
  static const size_t ratiosSize = C::MAXSAMPLES*(C::MAXSAMPLES-1)/2;

  Scalar  amplitudes_[amplitudesSize];
  Scalar  amplitudeErrors_[amplitudesSize];
  Scalar  amplitudeErrors2_[amplitudesSize];
  Scalar  amplitudeErrors2inv_[amplitudesSize];
 
  std::vector < Ratio > ratios_;
  std::vector < Tmax > times_;
  std::vector < Tmax > timesAB_;
  
  Scalar pedestal_;
  int    num_;
  Scalar ampMaxError_;
  
  CalculatedRecHit calculatedRechit_;
};

template <class C, typename Scalar>
void EcalUncalibRecHitRatioMethodAlgo<C,Scalar>::init( const C &dataFrame, const InputScalar * pedestals, const InputScalar * pedestalRMSes, const InputScalar * gainRatios )
{
  calculatedRechit_.timeMax = 5;
  calculatedRechit_.amplitudeMax = 0;
  calculatedRechit_.timeError = -999;

  ratios_.clear();
  ratios_.reserve(C::MAXSAMPLES*(C::MAXSAMPLES-1)/2);
  times_.clear();
  times_.reserve(C::MAXSAMPLES*(C::MAXSAMPLES-1)/2);
  timesAB_.clear();
  timesAB_.reserve(C::MAXSAMPLES*(C::MAXSAMPLES-1)/2);
  
  // to obtain gain 12 pedestal:
  // -> if it's in gain 12, use first sample
  // --> average it with second sample if in gain 12 and 3-sigma-noise compatible (better LF noise cancellation)
  // -> else use pedestal from database
  pedestal_ = 0;
  num_      = 0;
  if (dataFrame.sample(0).gainId() == 1) {
    pedestal_ += Scalar (dataFrame.sample(0).adc());
    num_++;
  }
  if (num_!=0 &&
      dataFrame.sample(1).gainId() == 1 && 
      fabs(dataFrame.sample(1).adc()-dataFrame.sample(0).adc())<3*pedestalRMSes[0]) {
    pedestal_ += Scalar (dataFrame.sample(1).adc());
    num_++;
  }
  if (num_ != 0)
    pedestal_ /= num_;
  else
    pedestal_ = pedestals[0];
  
  // fill vector of amplitudes, pedestal subtracted and vector
  // of amplitude uncertainties Also, find the uncertainty of a
  // sample with max amplitude. We will use it later.
  
  ampMaxError_ = 0;
  Scalar ampMaxValue = -1000;
  
  // ped-subtracted and gain-renormalized samples. It is VERY
  // IMPORTANT to have samples one clock apart which means to
  // have vector size equal to MAXSAMPLES
  Scalar sample;
  Scalar sampleError;
  int GainId;
  for (int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
    GainId = dataFrame.sample(iSample).gainId();
    
    if (GainId == 1) {
      sample      = Scalar (dataFrame.sample(iSample).adc() - pedestal_);
      sampleError = pedestalRMSes[0];
    } else if (GainId == 2 || GainId == 3){
      sample      = (Scalar (dataFrame.sample(iSample).adc() - pedestals[GainId - 1])) *gainRatios[GainId - 1];
      sampleError = pedestalRMSes[GainId-1]*gainRatios[GainId-1];
    } else {
      sample      = 1e-9;  // GainId=0 case falls here, from saturation
      sampleError = 1e+9;  // inflate error so won't generate ratio considered for the measurement 
    }
    
  
    if(sampleError>0){
      amplitudes_[iSample]=sample;
      amplitudeErrors_[iSample]=sampleError;
      if(ampMaxValue < sample){
	ampMaxValue = sample;
	ampMaxError_ = sampleError;
      }
    }else{
      // inflate error for useless samples
      amplitudes_[iSample]=sample;
      amplitudeErrors_[iSample]=1e+9;
    }
    amplitudeErrors2_[iSample] =  amplitudeErrors_[iSample]*amplitudeErrors_[iSample];
    amplitudeErrors2inv_[iSample] = 1.0/amplitudeErrors2_[iSample];
  }

}

template<class C, typename Scalar>
void EcalUncalibRecHitRatioMethodAlgo<C,Scalar>::computeAmpChi2(Scalar sumAA, Scalar t, Scalar alpha, Scalar overab, Scalar & chi2, Scalar & amp) const {
  Scalar sumAf = 0;
  Scalar sumff = 0;
  for(unsigned int it = 0; it < amplitudesSize; it++){
    Scalar err2 = amplitudeErrors2inv_[it];
    Scalar offset = (Scalar(it) - t)*overab;
    Scalar term1 = 1.0 + offset;
    if(term1>1e-6){
      Scalar f = std::exp( alpha*(std::log(term1) - offset) );
      sumAf += amplitudes_[it]*(f*err2);
      sumff += f*(f*err2);
    }
  }
  
  chi2 = sumAA;
  amp = 0.;
  if( sumff > 0 ){
    amp = sumAf/sumff;
    chi2 = sumAA - sumAf*amp;
  }
  chi2 /= Scalar( amplitudesSize);
}

template<class C, typename Scalar>
void EcalUncalibRecHitRatioMethodAlgo<C,Scalar>::computeTime(std::vector < InputScalar >&timeFitParameters,
	    std::pair < InputScalar, InputScalar >&timeFitLimits, std::vector < InputScalar >&amplitudeFitParameters)
{
  //////////////////////////////////////////////////////////////
  //                                                          //
  //              RATIO METHOD FOR TIME STARTS HERE           //
  //                                                          //
  //////////////////////////////////////////////////////////////
  Scalar ampMaxAlphaBeta = 0;
  Scalar tMaxAlphaBeta = 5;
  Scalar tMaxErrorAlphaBeta = 999;
  Scalar tMaxRatio = 5;
  Scalar tMaxErrorRatio = 999;

  Scalar sumAA = 0;
  Scalar sumA  = 0;
  Scalar sum1  = 0;
  Scalar sum0  =  amplitudesSize;
  Scalar NullChi2 = 0;

  // null hypothesis = no pulse, pedestal only
  for(unsigned int i = 0; i < amplitudesSize; i++){
    Scalar err2 = amplitudeErrors2inv_[i];
    //sum0  += 1;
    sum1  += err2;
    sumA  += (amplitudes_[i]*err2);
    sumAA += amplitudes_[i]*(amplitudes_[i]*err2);
  }
  if(sum0>0){
    NullChi2 = (sumAA - sumA*sumA/sum1)/sum0;
  }else{
    // not enough samples to reconstruct the pulse
    return;
  }

  // Make all possible Ratio's based on any pair of samples i and j
  // (j>i) with positive amplitudes_
  //
  //       Ratio[k] = Amp[i]/Amp[j]
  //       where Amp[i] is pedestal subtracted ADC value in a time sample [i]
  //
  Scalar alphabeta = amplitudeFitParameters[0]*amplitudeFitParameters[1];
  Scalar alpha = amplitudeFitParameters[0];
  Scalar beta = amplitudeFitParameters[1];
  Scalar overab = 1.0/alphabeta;
  Scalar RLimits[amplitudesSize];
  for(unsigned int i = 1; i < amplitudesSize; ++i){
    RLimits[i] = exp(Scalar(i)/beta)-0.001;
  } 
  Scalar stat=1;  // pedestal from db
  if(num_>0) stat =  1.0/sqrt(Scalar(num_));      // num presampeles used to compute pedestal

  for(unsigned int i = 0; i < amplitudesSize-1; i++){
    for(unsigned int j = i+1; j < amplitudesSize; j++){

      if(amplitudes_[i]>1 && amplitudes_[j]>1){

	// ratio
	Scalar Rtmp = amplitudes_[i]/amplitudes_[j];

	// don't include useless ratios
	if( Rtmp<0.001 ||  Rtmp> RLimits[j-i] ) continue;

	// error^2 due to stat fluctuations of time samples
	// (uncorrelated for both samples)

	Scalar err1 = Rtmp*Rtmp*( (amplitudeErrors2_[i]/(amplitudes_[i]*amplitudes_[i])) + (amplitudeErrors2_[j]/(amplitudes_[j]*amplitudes_[j])) );

	// error due to fluctuations of pedestal (common to both samples)
	Scalar err2 = stat*amplitudeErrors_[j]*(amplitudes_[i]-amplitudes_[j])/(amplitudes_[j]*amplitudes_[j]);

	//error due to integer round-down. It is relevant to low
	//amplitudes_ in gainID=1 and negligible otherwise.
        Scalar err3 = 0.289/amplitudes_[j];

	Scalar totalError = sqrt(err1 + err2*err2 +err3*err3);


	// don't include useless ratios
	if(totalError < 1.0
	   && Rtmp>0.001
	   && Rtmp< RLimits[j-i]
	   ){
	  Ratio currentRatio = { i, (j-i), Rtmp, totalError };
	  ratios_.push_back(currentRatio);
	}

      }

    }
  }

  // No useful ratios, return zero amplitude and no time measurement
  if(!ratios_.size() >0)
    return;

  // make a vector of Tmax measurements that correspond to each ratio
  // and based on Alpha-Beta parameterization of the pulse shape

  for(unsigned int i = 0; i < ratios_.size(); i++){

    Scalar stepOverBeta = Scalar(ratios_[i].step)/beta;
    Scalar offset = Scalar(ratios_[i].index) + alphabeta;

    Scalar Rmin = ratios_[i].value - ratios_[i].error;
    if(Rmin<0.001) Rmin=0.001;

    Scalar Rmax = ratios_[i].value + ratios_[i].error;
    if( Rmax > RLimits[ratios_[i].step] ) Rmax = RLimits[ratios_[i].step];

    // real time is offset - timeN
    Scalar time1 = ratios_[i].step/(exp((stepOverBeta-log(Rmin))/alpha)-1.0);
    Scalar time2 = ratios_[i].step/(exp((stepOverBeta-log(Rmax))/alpha)-1.0);

    // this is the time measurement based on the ratios[i]
    Scalar tmax = offset - 0.5 * (time1 + time2);
    Scalar tmaxerr = 0.5 * std::abs(time1 - time2);

    // calculate chi2
    Scalar chi2=0.;
    Scalar amp=0.;
    computeAmpChi2(sumAA, tmax,alpha, overab, chi2,amp);

    // choose reasonable measurements. One might argue what is
    // reasonable and what is not.
    if(chi2 > 0 && tmaxerr > 0 && tmax > 0){
      Tmax currentTmax={ ratios_[i].index, ratios_[i].step, tmax, tmaxerr, amp, chi2 };
      timesAB_.push_back(currentTmax);
    }
  }

  // no reasonable time measurements!
  if( !(timesAB_.size()> 0))
    return;

  // find minimum chi2
  Scalar chi2min = 1.0e+9;
  Scalar timeMinimum = 5;
  Scalar errorMinimum = 999;
  for(unsigned int i = 0; i < timesAB_.size(); i++){
    if( timesAB_[i].chi2 <= chi2min ){
      chi2min = timesAB_[i].chi2;
      timeMinimum = timesAB_[i].value;
      errorMinimum = timesAB_[i].error;
    }
  }

  // make a weighted average of tmax measurements with "small" chi2
  // (within 1 sigma of statistical uncertainty :-)
  Scalar chi2Limit = chi2min + 1.0;
  Scalar time_max = 0;
  Scalar time_wgt = 0;
  for(unsigned int i = 0; i < timesAB_.size(); i++){
    if(  timesAB_[i].chi2 < chi2Limit  ){
      Scalar inverseSigmaSquared = 1.0/(timesAB_[i].error*timesAB_[i].error);
      time_wgt += inverseSigmaSquared;
      time_max += timesAB_[i].value*inverseSigmaSquared;
    }
  }

  tMaxAlphaBeta =  time_max/time_wgt;
  tMaxErrorAlphaBeta = 1.0/sqrt(time_wgt);

  Scalar chi2AlphaBeta = 0.;
  // find amplitude and chi2
  computeAmpChi2(sumAA, tMaxAlphaBeta, alpha, overab, chi2AlphaBeta, ampMaxAlphaBeta);

  if(ampMaxAlphaBeta==0 || chi2AlphaBeta > NullChi2)
    // // no visible pulse here or null hypothesis is better
    return;
  

 

  // if we got to this point, we have a reconstructied Tmax
  // using RatioAlphaBeta Method. To summarize:
  //
  //     tMaxAlphaBeta      - Tmax value
  //     tMaxErrorAlphaBeta - error on Tmax, but I would not trust it
  //     ampMaxAlphaBeta    - amplitude of the pulse
  //     ampMaxError_        - uncertainty of the time sample with max amplitude
  //



  // Do Ratio's Method with "large" pulses
  if( ampMaxAlphaBeta > 5.0*ampMaxError_ ){

    // make a vector of Tmax measurements that correspond to each
    // ratio. Each measurement have it's value and the error
    
    Scalar time_max = 0;
    Scalar time_wgt = 0;
    
    
    for (unsigned int i = 0; i < ratios_.size(); i++) {
      
      if(ratios_[i].step == 1
	 && ratios_[i].value >= timeFitLimits.first
	 && ratios_[i].value <= timeFitLimits.second
	 ){
	
	Scalar time_max_i = ratios_[i].index;
	
	// calculate polynomial for Tmax
	
	Scalar u = timeFitParameters[timeFitParameters.size() - 1];
	for (int k = timeFitParameters.size() - 2; k >= 0; k--) {
	  u = u * ratios_[i].value + timeFitParameters[k];
	}
	
	// calculate derivative for Tmax error
	Scalar du =
	  (timeFitParameters.size() -
	   1) * timeFitParameters[timeFitParameters.size() - 1];
	for (int k = timeFitParameters.size() - 2; k >= 1; k--) {
	  du = du * ratios_[i].value + k * timeFitParameters[k];
	}
	
	
	// running sums for weighted average
	Scalar errorsquared =
	  (ratios_[i].error * du) * (ratios_[i].error * du);
	if (errorsquared > 0) {
	  Scalar oe =  1.0 / errorsquared;
	  time_max += oe * (time_max_i - u);
	  time_wgt += oe;
	  Tmax currentTmax =
	    { ratios_[i].index, 1, (time_max_i - u),
	      (ratios_[i].error * du) ,0,1 };
	  times_.push_back(currentTmax);
	  
	}
      }
    }
    
    
    // calculate weighted average of all Tmax measurements
    if (time_wgt > 0) {
      tMaxRatio = time_max/time_wgt;
      tMaxErrorRatio = 1.0/sqrt(time_wgt);
      
      // combine RatioAlphaBeta and Ratio Methods
      
      if( ampMaxAlphaBeta < 10.0*ampMaxError_  ){
	
	// use pure Ratio Method
	calculatedRechit_.timeMax = tMaxRatio;
	calculatedRechit_.timeError = tMaxErrorRatio;
	
      }else{
	
	// combine two methods
	calculatedRechit_.timeMax = 0.2*( tMaxAlphaBeta*(10.0-(ampMaxAlphaBeta/ampMaxError_)) +
				      tMaxRatio*((ampMaxAlphaBeta/ampMaxError_) - 5.0) );
	calculatedRechit_.timeError = 0.2*( tMaxErrorAlphaBeta*(10.0-(ampMaxAlphaBeta/ampMaxError_)) + 
					tMaxErrorRatio*((ampMaxAlphaBeta/ampMaxError_) - 5.0) );
	
      }
      
    }else{
      
      // use RatioAlphaBeta Method
      calculatedRechit_.timeMax = tMaxAlphaBeta;
      calculatedRechit_.timeError = tMaxErrorAlphaBeta;
      
    }
    
  }else{

    // use RatioAlphaBeta Method
    calculatedRechit_.timeMax = tMaxAlphaBeta;
    calculatedRechit_.timeError = tMaxErrorAlphaBeta;
    
  }
}

template<class C, typename Scalar>
void EcalUncalibRecHitRatioMethodAlgo<C,Scalar>::computeAmplitude( std::vector< InputScalar > &amplitudeFitParameters )
{
	////////////////////////////////////////////////////////////////
	//                                                            //
	//             CALCULATE AMPLITUDE                            //
	//                                                            //
	////////////////////////////////////////////////////////////////


	Scalar alpha = amplitudeFitParameters[0];
	Scalar beta = amplitudeFitParameters[1];

	// calculate pedestal, again

	Scalar pedestalLimit = calculatedRechit_.timeMax - (alpha * beta) - 1.0;
	Scalar sumA = 0;
	Scalar sumF = 0;
	Scalar sumAF = 0;
	Scalar sumFF = 0;
	Scalar sum1 = 0;
	for (unsigned int i = 0; i < amplitudesSize; i++) {
	  Scalar err2 = amplitudeErrors2inv_[i];
	  Scalar f = 0;
	  Scalar termOne = 1 + (i - calculatedRechit_.timeMax) / (alpha * beta);
	  if (termOne > 1.e-5) f = std::exp(alpha * std::log(termOne) -(i - calculatedRechit_.timeMax) / beta);
	  
	  // apply range of interesting samples
	  
	  if ( (i < pedestalLimit)
	       || (f > 0.6 && i <= calculatedRechit_.timeMax)
	       || (f > 0.4 && i >= calculatedRechit_.timeMax)) {
	    sum1  += err2;
	    sumA  += amplitudes_[i]*err2;
	    sumF  += (f*err2);
	    sumAF += amplitudes_[i]*(f*err2);
	    sumFF += f*(f*err2);
	  }
	}
	
	calculatedRechit_.amplitudeMax = 0;
	if(sum1 > 0){
	  Scalar denom = sumFF*sum1 - sumF*sumF;
	  if(fabs(denom)>1.0e-20){
	    calculatedRechit_.amplitudeMax = (sumAF*sum1 - sumA*sumF)/denom;
	  }
	}
}



  template < class C, typename Scalar > EcalUncalibratedRecHit
  EcalUncalibRecHitRatioMethodAlgo < C, Scalar >::makeRecHit(const C & dataFrame,
						       const InputScalar *pedestals,
                                                       const InputScalar *pedestalRMSes,
						       const InputScalar *gainRatios,
						       std::vector < InputScalar >&timeFitParameters,
						       std::vector < InputScalar >&amplitudeFitParameters,
						       std::pair < InputScalar, InputScalar >&timeFitLimits)
{

        init( dataFrame, pedestals, pedestalRMSes, gainRatios );
        computeTime( timeFitParameters, timeFitLimits, amplitudeFitParameters );
        computeAmplitude( amplitudeFitParameters );

	// 1st parameters is id
	//
	// 2nd parameters is amplitude. It is calculated by this method.
	//
	// 3rd parameter is pedestal. It is not calculated. This method
	// relies on input parameters for pedestals and gain ratio. Return
	// zero.
	//
	// 4th parameter is jitter which is a bad choice to call Tmax. It is
	// calculated by this method (in 25 nsec clock units)
	//
	// GF subtract 5 so that jitter==0 corresponds to synchronous hit
	//
	//
	// 5th parameter is chi2. It is possible to calculate chi2 for
	// Tmax. It is possible to calculate chi2 for Amax. However, these
	// values are not very useful without TmaxErr and AmaxErr. This
	// method can return one value for chi2 but there are 4 different
	// parameters that have useful information about the quality of Amax
	// ans Tmax. For now we can return TmaxErr. The quality of Tmax and
	// Amax can be judged from the magnitude of TmaxErr

	return EcalUncalibratedRecHit(dataFrame.id(),
				      calculatedRechit_.amplitudeMax,
                                      pedestal_,
				      calculatedRechit_.timeMax - 5,
				      calculatedRechit_.timeError);
}
#endif
