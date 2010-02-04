#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecChi2Algo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecChi2Algo_HH

/** \class EcalUncalibRecHitRecChi2Algo
  *
  *  Template used to compute the chi2 of an MGPA pulse for in-time and out-of-time signals, algorithm based on the chi2express.  
  *  The in-time chi2 is calculated against the time intercalibrations from the DB while the out-of-time chi2 is calculated
  *  against the Tmax measurement on event by event basis.
  *
  *  \author Konstantinos Theofilatos 02 Feb 2010 
  */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"
#include "CondFormats/EcalObjects/interface/EcalTimeCalibConstants.h"


#include <vector>

template<class C> class EcalUncalibRecHitRecChi2Algo 
{
  public:

  // destructor
  virtual ~EcalUncalibRecHitRecChi2Algo<C>() { };

  EcalUncalibRecHitRecChi2Algo<C>() { };
  EcalUncalibRecHitRecChi2Algo<C>(
				  const C& dataFrame,
				  const double amplitude,
				  const EcalTimeCalibConstant& timeIC,
				  const double amplitudeOutOfTime,
				  const double timePulse,
				  const double* pedestals,
				  const double* pedestalsRMS,
				  const double* gainRatios,
				  const EcalShapeBase & testbeamPulseShape
  );


  virtual float chi2(){return chi2_;}
  virtual float chi2OutOfTime(){return chi2OutOfTime_;}


  private:
    double chi2_;
    double chi2OutOfTime_;


};


template <class C>
EcalUncalibRecHitRecChi2Algo<C>::EcalUncalibRecHitRecChi2Algo(
			const C& dataFrame, 
			const double amplitude, 
			const EcalTimeCalibConstant& timeIC, 
			const double amplitudeOutOfTime, 
			const double timePulse, 
			const double* pedestals, 
			const double* pedestalsRMS,
			const double* gainRatios,
			const EcalShapeBase & testbeamPulseShape
) 
{
    chi2_=0;
    chi2OutOfTime_=0;
    double dynamicPedestal=0;
    double dynamicPedestalForEarlyPulse=0;


    int gainId0 = 1;
    int iGainSwitch = 0;
    bool isSaturated = 0;
    for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) // if gain switch use later the pedestal RMS, otherwise we use the pedestal from the DB
    {
      int gainId = dataFrame.sample(iSample).gainId();
      if ( gainId == 0 )
        {
          gainId = 3;
          isSaturated = 1;
        }
      if (gainId != gainId0) iGainSwitch = 1;

      if(gainId==1 && iSample<3)dynamicPedestal += 0.33333*dataFrame.sample(iSample).adc(); // calculate dynamic pedestal from first 3 presamples
      if(gainId==1 && iSample==0)dynamicPedestalForEarlyPulse = dataFrame.sample(iSample).adc(); // take only first presample to estimate the pedestal
      
    }


    // compute testbeamPulseShape shape parameters
    double ADC_clock = 25; // 25 ns
    double risingTime = testbeamPulseShape.timeToRise();
    double tzero = risingTime  - 5*ADC_clock;  // 5 samples before the peak

    double shiftTime = + timeIC; // we put positive here
    double shiftTimeOutOfTime = -timePulse; // we put negative here
    double pulseShape[10];
    double pulseShapeShifted[10];

    for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++)
    {
        int gainId = dataFrame.sample(iSample).gainId();

        if(gainId==0)continue; // skip saturated samples
        if(iSample==3 || iSample==9) continue; // don't include samples which were not used in the amplitude calculation

        double Si = double(dataFrame.sample(iSample).adc());

        pulseShape[iSample] = (testbeamPulseShape)(tzero + shiftTime + iSample*ADC_clock);
        double eventPedestal = !iGainSwitch ? dynamicPedestal:pedestals[gainId-1];  // use dynamic pedestal for G12 and average pedestal for G6,G1
        double diff = pulseShape[iSample]*amplitude  - (Si- eventPedestal)*gainRatios[gainId-1]; 
        chi2_+= (diff*diff)/(pedestalsRMS[gainId-1]*pedestalsRMS[gainId-1]);

        pulseShapeShifted[iSample] = (testbeamPulseShape)(tzero + shiftTimeOutOfTime + iSample*ADC_clock); // calculate out of time chi2
        double eventPedestalForEarlyPulse = !iGainSwitch ? dynamicPedestalForEarlyPulse:pedestals[gainId-1];  
        double diffOutOfTime = pulseShapeShifted[iSample]*amplitudeOutOfTime  - (Si- eventPedestalForEarlyPulse)*gainRatios[gainId-1]; 
        chi2OutOfTime_ += (diffOutOfTime*diffOutOfTime)/(pedestalsRMS[gainId-1]*pedestalsRMS[gainId-1]);
    }

}

#endif
