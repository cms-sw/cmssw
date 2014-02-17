#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRatioMethodAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRatioMethodAlgo_HH

/** \class EcalUncalibRecHitRatioMethodAlgo
 *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
 *  using a ratio method
 *
 *  $Id: EcalUncalibRecHitRatioMethodAlgo.h,v 1.50 2012/06/11 21:02:13 wmtan Exp $
 *  $Date: 2012/06/11 21:02:13 $
 *  $Revision: 1.50 $
 *  \author A. Ledovskoy (Design) - M. Balazs (Implementation)
 */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalSampleMask.h"
#include <vector>

template < class C > class EcalUncalibRecHitRatioMethodAlgo {
      public:
	struct Ratio {
		unsigned int index;
                unsigned int step;
		double value;
		double error;
	};
	struct Tmax {
		unsigned int index;
                unsigned int step;
		double value;
		double error;
                double amplitude;
                double chi2;
	};
	struct CalculatedRecHit {
		double amplitudeMax;
		double timeMax;
		double timeError;
                double chi2;
	};

	virtual ~ EcalUncalibRecHitRatioMethodAlgo < C > () { };
	virtual EcalUncalibratedRecHit makeRecHit(const C & dataFrame,
						  const EcalSampleMask & sampleMask,
						  const double *pedestals,
                                                  const double* pedestalRMSes,
						  const double *gainRatios,
						  std::vector < double >&timeFitParameters,
						  std::vector < double >&amplitudeFitParameters,
						  std::pair < double, double >&timeFitLimits);

        // more function to be able to compute
        // amplitude and time separately
        void init( const C &dataFrame, const EcalSampleMask &sampleMask, const double * pedestals, const double * pedestalRMSes, const double * gainRatios );
        void computeTime(std::vector < double >&timeFitParameters, std::pair < double, double >&timeFitLimits, std::vector< double > &amplitudeFitParameters);
        void computeAmplitude( std::vector< double > &amplitudeFitParameters );
        CalculatedRecHit getCalculatedRecHit() { return calculatedRechit_; };
	bool fixMGPAslew( const C &dataFrame );

      protected:
	
        EcalSampleMask sampleMask_;
	DetId          theDetId_;
	std::vector < double > amplitudes_;
        std::vector < double > amplitudeErrors_;
	std::vector < Ratio > ratios_;
	std::vector < Tmax > times_;
        std::vector < Tmax > timesAB_;

        double pedestal_;
	int    num_;
        double ampMaxError_;

	CalculatedRecHit calculatedRechit_;
};

template <class C>
void EcalUncalibRecHitRatioMethodAlgo<C>::init( const C &dataFrame, const EcalSampleMask &sampleMask, 
						const double * pedestals, const double * pedestalRMSes, const double * gainRatios )
{
        sampleMask_ = sampleMask;
	theDetId_ = DetId(dataFrame.id().rawId());  

	calculatedRechit_.timeMax = 5;
	calculatedRechit_.amplitudeMax = 0;
	calculatedRechit_.timeError = -999;
	amplitudes_.clear();
	amplitudes_.reserve(C::MAXSAMPLES);
	amplitudeErrors_.clear();
	amplitudeErrors_.reserve(C::MAXSAMPLES);
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
	if (dataFrame.sample(0).gainId() == 1 &&
	    sampleMask_.useSample(0, theDetId_ ) 
	    ) {
	  pedestal_ += double (dataFrame.sample(0).adc());
	  num_++;
	}
	if (num_!=0 &&
	    dataFrame.sample(1).gainId() == 1 && 
	    sampleMask_.useSample(1, theDetId_) &&
	    fabs(dataFrame.sample(1).adc()-dataFrame.sample(0).adc())<3*pedestalRMSes[0]) {
	        pedestal_ += double (dataFrame.sample(1).adc());
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
        double ampMaxValue = -1000;

	// ped-subtracted and gain-renormalized samples. It is VERY
	// IMPORTANT to have samples one clock apart which means to
	// have vector size equal to MAXSAMPLES
	double sample;
        double sampleError;
	int GainId;
        for (int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
	  


          GainId = dataFrame.sample(iSample).gainId();
	  
	  
	  // only use normally samples which are desired; if sample not to be used
	  // inflate error so won't generate ratio considered for the measurement
	  if (!sampleMask_.useSample(iSample, theDetId_ ) ) {
	    sample      = 1e-9;
	    sampleError = 1e+9;
	  }
          else if (GainId == 1) {
            sample      = double (dataFrame.sample(iSample).adc() - pedestal_);
            sampleError = pedestalRMSes[0];
          } 
	  else if (GainId == 2 || GainId == 3){
            sample      = (double (dataFrame.sample(iSample).adc() - pedestals[GainId - 1])) *gainRatios[GainId - 1];
            sampleError = pedestalRMSes[GainId-1]*gainRatios[GainId-1];
          } 
	  else {
	    sample      = 1e-9;  // GainId=0 case falls here, from saturation
	    sampleError = 1e+9;  // inflate error so won't generate ratio considered for the measurement 
	  }
	  

          if(sampleError>0){
            amplitudes_.push_back(sample);
            amplitudeErrors_.push_back(sampleError);
            if(ampMaxValue < sample){
              ampMaxValue = sample;
              ampMaxError_ = sampleError;
            }
          }else{
	    // inflate error for useless samples
            amplitudes_.push_back(sample);
            amplitudeErrors_.push_back(1e+9);
	  }
        }
}

template <class C>
bool EcalUncalibRecHitRatioMethodAlgo<C>::fixMGPAslew( const C &dataFrame )
{

  // This fuction finds sample(s) preceeding gain switching and
  // inflates errors on this sample, therefore, making this sample
  // invisible for Ratio Method. Only gain switching DOWN is
  // considered Only gainID=1,2,3 are considered. In case of the
  // saturation (gainID=0), we keep "distorted" sample because it is
  // the only chance to make time measurement; the qualilty of it will
  // be bad anyway.

  bool result = false;

  int GainIdPrev;
  int GainIdNext;
  for (int iSample = 1; iSample < C::MAXSAMPLES; iSample++) {

    // only use samples which are desired
    if (!sampleMask_.useSample(iSample, theDetId_) ) continue;
    
    GainIdPrev = dataFrame.sample(iSample-1).gainId();
    GainIdNext = dataFrame.sample(iSample).gainId();
    if( GainIdPrev>=1 && GainIdPrev<=3 && 
        GainIdNext>=1 && GainIdNext<=3 && 
        GainIdPrev<GainIdNext ){
      amplitudes_[iSample-1]=1e-9;
      amplitudeErrors_[iSample-1]=1e+9;
      result = true;      
    }
  }
  return result;

}

template<class C>
void EcalUncalibRecHitRatioMethodAlgo<C>::computeTime(std::vector < double >&timeFitParameters,
	    std::pair < double, double >&timeFitLimits, std::vector < double >&amplitudeFitParameters)
{
  //////////////////////////////////////////////////////////////
  //                                                          //
  //              RATIO METHOD FOR TIME STARTS HERE           //
  //                                                          //
  //////////////////////////////////////////////////////////////
  double ampMaxAlphaBeta = 0;
  double tMaxAlphaBeta = 5;
  double tMaxErrorAlphaBeta = 999;
  double tMaxRatio = 5;
  double tMaxErrorRatio = 999;

  double sumAA = 0;
  double sumA  = 0;
  double sum1  = 0;
  double sum0  = 0;
  double sumAf = 0;
  double sumff = 0;
  double NullChi2 = 0;

  // null hypothesis = no pulse, pedestal only
  for(unsigned int i = 0; i < amplitudes_.size(); i++){
    double err2 = amplitudeErrors_[i]*amplitudeErrors_[i];
    sum0  += 1;
    sum1  += 1/err2;
    sumA  += amplitudes_[i]/err2;
    sumAA += amplitudes_[i]*amplitudes_[i]/err2;
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
  double alphabeta = amplitudeFitParameters[0]*amplitudeFitParameters[1];
  double alpha = amplitudeFitParameters[0];
  double beta = amplitudeFitParameters[1];

  for(unsigned int i = 0; i < amplitudes_.size()-1; i++){
    for(unsigned int j = i+1; j < amplitudes_.size(); j++){

      if(amplitudes_[i]>1 && amplitudes_[j]>1){

	// ratio
	double Rtmp = amplitudes_[i]/amplitudes_[j];

	// error^2 due to stat fluctuations of time samples
	// (uncorrelated for both samples)

	double err1 = Rtmp*Rtmp*( (amplitudeErrors_[i]*amplitudeErrors_[i]/(amplitudes_[i]*amplitudes_[i])) + (amplitudeErrors_[j]*amplitudeErrors_[j]/(amplitudes_[j]*amplitudes_[j])) );

	// error due to fluctuations of pedestal (common to both samples)
	double stat;
	if(num_>0) stat = num_;      // num presampeles used to compute pedestal
	else       stat = 1;         // pedestal from db
	double err2 = amplitudeErrors_[j]*(amplitudes_[i]-amplitudes_[j])/(amplitudes_[j]*amplitudes_[j])/sqrt(stat);

	//error due to integer round-down. It is relevant to low
	//amplitudes_ in gainID=1 and negligible otherwise.
        double err3 = 0.289/amplitudes_[j];

	double totalError = sqrt(err1 + err2*err2 +err3*err3);


	// don't include useless ratios
	if(totalError < 1.0
	   && Rtmp>0.001
	   && Rtmp<exp(double(j-i)/beta)-0.001
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

    double stepOverBeta = double(ratios_[i].step)/beta;
    double offset = double(ratios_[i].index) + alphabeta;

    double Rmin = ratios_[i].value - ratios_[i].error;
    if(Rmin<0.001) Rmin=0.001;

    double Rmax = ratios_[i].value + ratios_[i].error;
    double RLimit = exp(stepOverBeta)-0.001;
    if( Rmax > RLimit ) Rmax = RLimit;

    double time1 = offset - ratios_[i].step/(exp((stepOverBeta-log(Rmin))/alpha)-1.0);
    double time2 = offset - ratios_[i].step/(exp((stepOverBeta-log(Rmax))/alpha)-1.0);

    // this is the time measurement based on the ratios[i]
    double tmax = 0.5 * (time1 + time2);
    double tmaxerr = 0.5 * sqrt( (time1 - time2)*(time1 - time2) );

    // calculate chi2
    sumAf = 0;
    sumff = 0;
    for(unsigned int it = 0; it < amplitudes_.size(); it++){
      double err2 = amplitudeErrors_[it]*amplitudeErrors_[it];
      double offset = (double(it) - tmax)/alphabeta;
      double term1 = 1.0 + offset;
      if(term1>1e-6){
	double f = exp( alpha*(log(1.0+offset) - offset) );
	sumAf += amplitudes_[it]*f/err2;
	sumff += f*f/err2;
      }
    }

    double chi2 = sumAA;
    double amp = 0;
    if( sumff > 0 ){
      chi2 = sumAA - sumAf*sumAf/sumff;
      amp = sumAf/sumff;
    }
    chi2 /= sum0;

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
  double chi2min = 1.0e+9;
  //double timeMinimum = 5;
  //double errorMinimum = 999;
  for(unsigned int i = 0; i < timesAB_.size(); i++){
    if( timesAB_[i].chi2 <= chi2min ){
      chi2min = timesAB_[i].chi2;
      //timeMinimum = timesAB_[i].value;
      //errorMinimum = timesAB_[i].error;
    }
  }

  // make a weighted average of tmax measurements with "small" chi2
  // (within 1 sigma of statistical uncertainty :-)
  double chi2Limit = chi2min + 1.0;
  double time_max = 0;
  double time_wgt = 0;
  for(unsigned int i = 0; i < timesAB_.size(); i++){
    if(  timesAB_[i].chi2 < chi2Limit  ){
      double inverseSigmaSquared = 1.0/(timesAB_[i].error*timesAB_[i].error);
      time_wgt += inverseSigmaSquared;
      time_max += timesAB_[i].value*inverseSigmaSquared;
    }
  }

  tMaxAlphaBeta =  time_max/time_wgt;
  tMaxErrorAlphaBeta = 1.0/sqrt(time_wgt);

  // find amplitude and chi2
  sumAf = 0;
  sumff = 0;
  for(unsigned int i = 0; i < amplitudes_.size(); i++){
    double err2 = amplitudeErrors_[i]*amplitudeErrors_[i];
    double offset = (double(i) - tMaxAlphaBeta)/alphabeta;
    double term1 = 1.0 + offset;
    if(term1>1e-6){
      double f = exp( alpha*(log(1.0+offset) - offset) );
      sumAf += amplitudes_[i]*f/err2;
      sumff += f*f/err2;
    }
  }

  if( sumff > 0 ){
    ampMaxAlphaBeta  = sumAf/sumff;
    double chi2AlphaBeta = (sumAA - sumAf*sumAf/sumff)/sum0;
    if(chi2AlphaBeta > NullChi2){
      // null hypothesis is better
      return;
    }

  }else{
    // no visible pulse here
    return;
  }

  // if we got to this point, we have a reconstructied Tmax
  // using RatioAlphaBeta Method. To summarize:
  //
  //     tMaxAlphaBeta      - Tmax value
  //     tMaxErrorAlphaBeta - error on Tmax, but I would not trust it
  //     ampMaxAlphaBeta    - amplitude of the pulse
  //     ampMaxError_        - uncertainty of the time sample with max amplitude
  //



  // Do Ratio's Method with "large" pulses
  if( ampMaxAlphaBeta/ampMaxError_ > 5.0 ){

	// make a vector of Tmax measurements that correspond to each
	// ratio. Each measurement have it's value and the error

	double time_max = 0;
	double time_wgt = 0;


	for (unsigned int i = 0; i < ratios_.size(); i++) {

          if(ratios_[i].step == 1
              && ratios_[i].value >= timeFitLimits.first
              && ratios_[i].value <= timeFitLimits.second
            ){

		double time_max_i = ratios_[i].index;

		// calculate polynomial for Tmax

		double u = timeFitParameters[timeFitParameters.size() - 1];
		for (int k = timeFitParameters.size() - 2; k >= 0; k--) {
			u = u * ratios_[i].value + timeFitParameters[k];
		}

		// calculate derivative for Tmax error
		double du =
		    (timeFitParameters.size() -
		     1) * timeFitParameters[timeFitParameters.size() - 1];
		for (int k = timeFitParameters.size() - 2; k >= 1; k--) {
			du = du * ratios_[i].value + k * timeFitParameters[k];
		}


		// running sums for weighted average
		double errorsquared =
		    ratios_[i].error * ratios_[i].error * du * du;
		if (errorsquared > 0) {

			time_max += (time_max_i - u) / errorsquared;
			time_wgt += 1.0 / errorsquared;
			Tmax currentTmax =
			    { ratios_[i].index, 1, (time_max_i - u),
		     sqrt(errorsquared),0,1 };
			times_.push_back(currentTmax);

		}
          }
        }


	// calculate weighted average of all Tmax measurements
	if (time_wgt > 0) {
          tMaxRatio = time_max/time_wgt;
          tMaxErrorRatio = 1.0/sqrt(time_wgt);

          // combine RatioAlphaBeta and Ratio Methods

          if( ampMaxAlphaBeta/ampMaxError_ > 10.0 ){

            // use pure Ratio Method
            calculatedRechit_.timeMax = tMaxRatio;
            calculatedRechit_.timeError = tMaxErrorRatio;

          }else{

            // combine two methods
            calculatedRechit_.timeMax = ( tMaxAlphaBeta*(10.0-ampMaxAlphaBeta/ampMaxError_) + tMaxRatio*(ampMaxAlphaBeta/ampMaxError_ - 5.0) )/5.0;
            calculatedRechit_.timeError = ( tMaxErrorAlphaBeta*(10.0-ampMaxAlphaBeta/ampMaxError_) + tMaxErrorRatio*(ampMaxAlphaBeta/ampMaxError_ - 5.0) )/5.0;

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

template<class C>
void EcalUncalibRecHitRatioMethodAlgo<C>::computeAmplitude( std::vector< double > &amplitudeFitParameters )
{
	////////////////////////////////////////////////////////////////
	//                                                            //
	//             CALCULATE AMPLITUDE                            //
	//                                                            //
	////////////////////////////////////////////////////////////////


	double alpha = amplitudeFitParameters[0];
	double beta = amplitudeFitParameters[1];

	// calculate pedestal, again

	double pedestalLimit = calculatedRechit_.timeMax - (alpha * beta) - 1.0;
	double sumA = 0;
	double sumF = 0;
	double sumAF = 0;
	double sumFF = 0;
	double sum1 = 0;
	for (unsigned int i = 0; i < amplitudes_.size(); i++) {
	        double err2 = amplitudeErrors_[i]*amplitudeErrors_[i];
		double f = 0;
		double termOne = 1 + (i - calculatedRechit_.timeMax) / (alpha * beta);
		if (termOne > 1.e-5) f = exp(alpha * log(termOne)) * exp(-(i - calculatedRechit_.timeMax) / beta);

		// apply range of interesting samples

		if ( (i < pedestalLimit)
		     || (f > 0.6 && i <= calculatedRechit_.timeMax)
		     || (f > 0.4 && i >= calculatedRechit_.timeMax)) {
		  	  sum1  += 1/err2;
			  sumA  += amplitudes_[i]/err2;
			  sumF  += f/err2;
			  sumAF += f*amplitudes_[i]/err2;
			  sumFF += f*f/err2;
		}
	}

	calculatedRechit_.amplitudeMax = 0;
	if(sum1 > 0){
	  double denom = sumFF*sum1 - sumF*sumF;
	  if(fabs(denom)>1.0e-20){
	    calculatedRechit_.amplitudeMax = (sumAF*sum1 - sumA*sumF)/denom;
	  }
	}
}



template < class C > EcalUncalibratedRecHit
    EcalUncalibRecHitRatioMethodAlgo < C >::makeRecHit(const C & dataFrame,
						       const EcalSampleMask & sampleMask,
						       const double *pedestals,
                                                       const double *pedestalRMSes,
						       const double *gainRatios,
						       std::vector < double >&timeFitParameters,
						       std::vector < double >&amplitudeFitParameters,
						       std::pair < double, double >&timeFitLimits)
{

        init( dataFrame, sampleMask, pedestals, pedestalRMSes, gainRatios );
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
