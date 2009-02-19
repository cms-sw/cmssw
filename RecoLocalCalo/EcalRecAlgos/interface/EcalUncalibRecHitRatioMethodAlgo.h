#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRatioMethodAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRatioMethodAlgo_HH

/** \class EcalUncalibRecHitRatioMethodAlgo
 *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
 *  using a ratio method
 *
 *  $Id: EcalUncalibRecHitRatioMethodAlgo.h,v 1.1 2009/02/19 22:16:18 franzoni Exp $
 *  $Date: 2009/02/19 22:16:18 $
 *  $Revision: 1.1 $
 *  \author A. Ledovskoy (Design) - M. Balazs (Implementation)
 */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include <vector>

template<class C> class EcalUncalibRecHitRatioMethodAlgo
{
 public:
  struct Ratio {int index; double value; double error;};
  struct Tmax  {int index; double value; double error;};
  struct CalculatedRechit {double amplitudeMax; double timeMax; double timeError;};

  virtual ~EcalUncalibRecHitRatioMethodAlgo<C>() {};
  virtual EcalUncalibratedRecHit makeRecHit(const C& dataFrame, const double* pedestals,
					    const double* gainRatios,
					    std::vector<double>& timeFitParameters,
					    std::vector<double>& amplitudeFitParameters,
					    std::pair<double,double>& timeFitLimits);

 protected:
  std::vector<double> amplitudes_;
  std::vector<Ratio> ratios_;
  std::vector<Tmax> times_;

};

template<class C> EcalUncalibratedRecHit
EcalUncalibRecHitRatioMethodAlgo<C>::makeRecHit(const C& dataFrame, const double* pedestals,
						const double* gainRatios,
						std::vector<double>& timeFitParameters,
						std::vector<double>& amplitudeFitParameters,
						std::pair<double,double>& timeFitLimits)
{
  CalculatedRechit calculatedRechit = {0,0,0};

  amplitudes_.clear();       amplitudes_.reserve(C::MAXSAMPLES);
  ratios_.clear();           ratios_.reserve(C::MAXSAMPLES);
  times_.clear();            times_.reserve(C::MAXSAMPLES);

  // make a vector of pedestal subtracted and gain corrected adc
  // values in each time sample.
  //
  // 1. Ideal situation: We rely on pedestal values from DB for each
  //    gain ID
  //
  // USE THIS ONE AS A DEFAULT

  /*
  for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
    int GainId = dataFrame.sample(iSample).gainId();
    double amplitude;

    if(GainId==1){amplitude = double(dataFrame.sample(iSample).adc())-pedestals[0];}
    else {amplitude = (double(dataFrame.sample(iSample).adc())-pedestals[GainId-1])*gainRatios[GainId-1];}

    amplitudes_.push_back(amplitude);

  }
  */

  //
  //  However, pedestal values from DB may differ from real pedestal
  //  values for this particular event. In this case
  //
  //  2. We can use first 1-3 time samples to estimate pedestal in
  //     gainID=1 if there is no pileup. For other gainID values we
  //     still need to rely on peestal values from DB
  //
  //  3. Use first time sample (or 1st, 2nd and 3rd) to estimate
  //     pedestal, compare it with pedestal value from DB for
  //     gainID==1, correct pedestal values for other gainIDs if there
  //     is gain switching in this event
  //
  // USE THIS ONE FOR TESTBEAM

  double pedestalCorrection = 0.0;

  // pedestal correction based on 1st datasample
  if(dataFrame.sample(0).gainId()==1){
    pedestalCorrection = double(dataFrame.sample(0).adc()) - pedestals[0];
  }

  for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
    int GainId = dataFrame.sample(iSample).gainId();
    double amplitude;

    if(GainId==1){
      amplitude = double(dataFrame.sample(iSample).adc()) - (pedestals[0]+pedestalCorrection);
    }else{
      amplitude = (double(dataFrame.sample(iSample).adc()) - (pedestals[GainId-1]+pedestalCorrection))*gainRatios[GainId-1];
    }

    amplitudes_.push_back(amplitude);

  }


  // Initial guess for Tmax using "Sum Of Three" method
  
  int tMax = -1;
  double SumOfThreeSamples = -100;
  for(unsigned int i=amplitudes_.size()-2; i>=1;  i--){
    double tmp = amplitudes_[i-1] + amplitudes_[i] + amplitudes_[i+1];
    if(tmp >= SumOfThreeSamples){
      SumOfThreeSamples = tmp;
      tMax = i;
    }
  }
  calculatedRechit.timeMax = double(tMax);
  double fraction = 0;
  if(SumOfThreeSamples>0) fraction = (amplitudes_[tMax+1]-amplitudes_[tMax-1])/SumOfThreeSamples;
  if(fraction>-1.0 && fraction<1.0) calculatedRechit.timeMax  = calculatedRechit.timeMax + fraction;
  calculatedRechit.amplitudeMax = SumOfThreeSamples/2.72;




  //////////////////////////////////////////////////////////////
  //                                                          //
  //              RATIO METHOD FOR TIME STARTS HERE           //
  //                                                          //
  //////////////////////////////////////////////////////////////



  // make a vector of Ratios and it's uncertainties
  //
  //       Ratio[i] = Amp[i]/Amp[i+1]
  //       where Amp[i] is pedestal subtracted ADC value in a time sample [i]
  //
  for(unsigned int i = 0; i < amplitudes_.size()-2; i++){

    if(amplitudes_[i]>0.1 && amplitudes_[i+1]>0.1){
      
      // effective amplitude for these two samples
      double Aeff = 1.0/sqrt( 1.0/(amplitudes_[i]*amplitudes_[i]) 
			      + 1.0/(amplitudes_[i+1]*amplitudes_[i+1]));

      // ratio 
      double Rtmp = amplitudes_[i]/amplitudes_[i+1];

      // error due to stat fluctuations of time samples
      double err1 = Rtmp/Aeff;

      // error due to fluctuations of pedestal
      double err2 = (1.0-Rtmp)/amplitudes_[i+1];

      if(    Aeff > 5.0 
	     && (i-calculatedRechit.timeMax)>-3.1 && (i-calculatedRechit.timeMax)<1.5
	     &&  Rtmp >= timeFitLimits.first && Rtmp <= timeFitLimits.second 
	     ){ 

	Ratio currentRatio = { i, Rtmp, sqrt( err1*err1 + err2*err2 ) };
	ratios_.push_back(currentRatio);

      }

    }

  }



  
  // make a vector of Tmax measurements that correspond to each
  // ratio. Each measurement have it's value and the error
  
  double time_max = 0;
  double time_wgt = 0;


  for(unsigned int i = 0; i < ratios_.size(); i++){

    double time_max_i = ratios_[i].index;

    // calculate polynomial for Tmax

    double u = timeFitParameters[timeFitParameters.size()-1];
    for(int k = timeFitParameters.size()-2; k >= 0; k--)
      u = u*ratios_[i].value + timeFitParameters[k];

    // calculate derivative for Tmax error
    double du = (timeFitParameters.size()-1)*timeFitParameters[timeFitParameters.size()-1];
    for(int k = timeFitParameters.size()-2; k >= 1; k--)
      du = du*ratios_[i].value + k*timeFitParameters[k];


    // running sums for weighted average
    double errorsquared = ratios_[i].error*ratios_[i].error*du*du;
    if(errorsquared > 0){

      time_max += (time_max_i - u)/errorsquared;
      time_wgt += 1.0/errorsquared;
      Tmax currentTmax={ ratios_[i].index, (time_max_i - u), sqrt(errorsquared) };
      times_.push_back(currentTmax);
      
    }
    
  }

  // calculate weighted average of all Tmax measurements
  if(time_wgt > 0){
    calculatedRechit.timeMax = time_max/time_wgt;
    calculatedRechit.timeError = 1.0/sqrt(time_wgt);
  }

  // calculate chisquared of all Tmax measurements. These are internal
  // values and they don't get returned

  double chi2one = 0;
  double chi2two = 0;
  for(unsigned int i=0; i<times_.size(); i++){
    double dummy = (times_[i].value - calculatedRechit.timeMax)/times_[i].error;
    chi2one += dummy*dummy;
    chi2two += (times_[i].value - calculatedRechit.timeMax)*(times_[i].value - calculatedRechit.timeMax);
  }
  if(times_.size()>1){
    chi2one = chi2one/(times_.size()-1);
    chi2two = chi2two/(times_.size()-1);
  }


  ////////////////////////////////////////////////////////////////
  //                                                            //
  //             CALCULATE AMPLITUDE                            //
  //                                                            //
  ////////////////////////////////////////////////////////////////


  double alpha = amplitudeFitParameters[0];
  double beta  = amplitudeFitParameters[1];

  double sum1 = 0;
  double sum2 = 0;
  for(unsigned int i = 0; i < amplitudes_.size()-1; i++){
  
    double f = 0;
    double termOne = 1 + (i-calculatedRechit.timeMax)/(alpha*beta);
    if(termOne>1.e-5) f =  exp(alpha*log(termOne))*exp(-(i-calculatedRechit.timeMax)/beta);
    if( (f>0.6 && i<=calculatedRechit.timeMax) || (f>0.4 && i>=calculatedRechit.timeMax) ){
      sum1 += amplitudes_[i]*f;
      sum2 += f*f;
    }

  }
  if(sum2>0) calculatedRechit.amplitudeMax = sum1/sum2;


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

  return EcalUncalibratedRecHit(dataFrame.id(),calculatedRechit.amplitudeMax,0,calculatedRechit.timeMax-5,calculatedRechit.timeError);
}

#endif
