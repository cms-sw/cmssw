#ifndef RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecWeightsAlgo_HH
#define RecoLocalCalo_EcalRecAlgos_EcalUncalibRecHitRecWeightsAlgo_HH

/** \class EcalUncalibRecHitRecWeightsAlgo
  *  Template used to compute amplitude, pedestal, time jitter, chi2 of a pulse
  *  using a weights method
  *
  *  $Id: EcalUncalibRecHitRecWeightsAlgo.h,v 1.9 2009/03/27 18:07:38 ferriff Exp $
  *  $Date: 2009/03/27 18:07:38 $
  *  $Revision: 1.9 $
  *  \author R. Bruneliere - A. Zabi
  *  
  *  
  *  The chi2 computation with matrix is replaced by the chi2express which is faster and provides correctly normalized chi2
  *  when gain switch and skips saturated samples. 
  *  : Kostas Theofilatos
  *  
  */

#include "Math/SVector.h"
#include "Math/SMatrix.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAbsAlgo.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
#include "SimCalorimetry/EcalSimAlgos/interface/EcalShapeBase.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include <vector>

template<class C> class EcalUncalibRecHitRecWeightsAlgo 
{
 public:
  // destructor
  virtual ~EcalUncalibRecHitRecWeightsAlgo<C>() { };

  /// Compute parameters
   virtual EcalUncalibratedRecHit makeRecHit(
					      const C& dataFrame 
					      , const double* pedestals
					      , const double* pedestalsRMS
					      , const double* gainRatios 
					      , const EcalWeightSet::EcalWeightMatrix** weights
					      , const EcalShapeBase & testbeamPulseShape
    ) {
    double amplitude_(-1.),  pedestal_(-1.), jitter_(-1.), chi2_(-1.);
    uint32_t flag = 0;


    // Get time samples
    ROOT::Math::SVector<double,C::MAXSAMPLES> frame;
    int gainId0 = 1;
    int iGainSwitch = 0;
    bool isSaturated = 0;
    for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++) {
      int gainId = dataFrame.sample(iSample).gainId();
      //Handling saturation (treating saturated gainId as maximum gain)
      if ( gainId == 0 ) 
	{ 
	  gainId = 3;
	  isSaturated = 1;
	}

      if (gainId != gainId0) iGainSwitch = 1;
      if (!iGainSwitch)
	frame(iSample) = double(dataFrame.sample(iSample).adc());
      else
	frame(iSample) = double(((double)(dataFrame.sample(iSample).adc()) - pedestals[gainId-1]) * gainRatios[gainId-1]);
    }

    // Compute parameters
    ROOT::Math::SVector <double,3> param = (*(weights[iGainSwitch])) * frame;
    amplitude_ = param(EcalUncalibRecHitRecAbsAlgo<C>::iAmplitude);
    pedestal_ = param(EcalUncalibRecHitRecAbsAlgo<C>::iPedestal);
    if (amplitude_) jitter_ = param(EcalUncalibRecHitRecAbsAlgo<C>::iTime);


	///////////////////////////////////////
       ///////// compute chi2express /////////   
      ///////////////////////////////////////

    // compute testbeamPulseShape shape parameters
    double ADC_clock = 25; // 25 ns
    double risingTime = testbeamPulseShape.timeToRise();
    double tzero = risingTime  - 5*ADC_clock;  // 5 samples before the peak
    double shiftTime=0;
    double pulseShape[10];

    double chi2express=0;  // initialized always to 0
    for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++)
    {
        int gainId = dataFrame.sample(iSample).gainId();
        if(gainId==0)continue; // skip saturated samples
        if(gainId>2 || (gainId-1)<0)  // safety check (array size)
        {
            edm::LogError("EcalUncalibRecHitRecAbsAlgo") << "gainId has not allowded value = "<<gainId;
            continue;
        }

        pulseShape[iSample] = (testbeamPulseShape)(tzero + shiftTime + iSample*ADC_clock);

        if(iSample==3 || iSample==9) continue; // don't include samples which were not used in the amplitude calculation

        double Si = double(dataFrame.sample(iSample).adc());
        double eventPedestal = !iGainSwitch ? pedestal_:pedestals[gainId-1];  // use event pedestal for G12 and average pedestal for G6,G1
        double Diff = pulseShape[iSample]*amplitude_  - (Si- eventPedestal)*gainRatios[gainId-1]; //
        chi2express+= (Diff*Diff)/(pedestalsRMS[gainId-1]*pedestalsRMS[gainId-1]);
    }
    chi2_ = chi2express;

      ////////////////////////////////////////////////////////////////
     ////////////////// compute max sample index ////////////////////
    ////////////////////////////////////////////////////////////////


    int maxSampleIndex=-1; 
    double maxSampleValue=0;  
    double maxSampleGainRatio=0;

    for(int iSample = 0; iSample < C::MAXSAMPLES; iSample++)
    {
 	int gainId = dataFrame.sample(iSample).gainId();
        if(gainId>2 || (gainId-1)<0)  // safety check (array size)
        {
            edm::LogError("EcalUncalibRecHitRecAbsAlgo") << "gainId has not allowded value = "<<gainId;
            continue;
        }

      if(gainId!=0)
      {
	double Si = dataFrame.sample(iSample).adc();
	double gratio = gainRatios[gainId-1];
	if(Si*gratio>maxSampleValue*maxSampleGainRatio)
	{
	    maxSampleValue=Si;
	    maxSampleGainRatio=gratio;
	    maxSampleIndex=iSample;
	}
      }else{maxSampleIndex=iSample;maxSampleGainRatio=-1;break;} // returns first saturated sample
    }
    jitter_ = double(maxSampleIndex);



    //When saturated gain flag i
    if (isSaturated)
      {
        flag = EcalUncalibratedRecHit::kSaturated;
	amplitude_ = double((4095. - pedestals[2]) * gainRatios[2]);
      }
    return EcalUncalibratedRecHit( dataFrame.id(), amplitude_, pedestal_, jitter_, chi2_, flag);
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
};
#endif
