//---------------------------------------------------------------------------
#ifndef HBHE_PULSESHAPE_FLAG_H_IKAJHGEWRHIGKHAWFIKGHAWIKGH
#define HBHE_PULSESHAPE_FLAG_H_IKAJHGEWRHIGKHAWFIKGHAWIKGH
//---------------------------------------------------------------------------
// Fitting-based algorithms for HBHE noise flagging
// 
// Included:
//    1. Linear discriminator (chi2 from linear fit / chi2 from nominal fit)
//    2. RMS8/Max ((RMS8/Max)^2 / chi2 from nominal fit)
//    3. Triangle fit
//
// Original Author: Yi Chen (Caltech), 6351 (Nov. 8, 2010)
//---------------------------------------------------------------------------
#include <string>
#include <vector>
#include <map>
//---------------------------------------------------------------------------
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibFormats/CaloObjects/interface/CaloSamples.h"
//---------------------------------------------------------------------------
class HBHEPulseShapeFlagSetter;
struct TriangleFitResult;
//---------------------------------------------------------------------------
class HBHEPulseShapeFlagSetter
{
public:
   HBHEPulseShapeFlagSetter(double MinimumChargeThreshold,
                            double TS4TS5ChargeThreshold,
                            double TS3TS4ChargeThreshold,
                            double TS3TS4UpperChargeThreshold,
                            double TS5TS6ChargeThreshold,
                            double TS5TS6UpperChargeThreshold,
                            double R45PlusOneRange,
                            double R45MinusOneRange,
			    unsigned int TrianglePeakTS,
			    const std::vector<double>& LinearThreshold, 
			    const std::vector<double>& LinearCut,
			    const std::vector<double>& RMS8MaxThreshold,
			    const std::vector<double>& RMS8MaxCut,
			    const std::vector<double>& LeftSlopeThreshold, 
			    const std::vector<double>& LeftSlopeCut,
			    const std::vector<double>& RightSlopeThreshold, 
			    const std::vector<double>& RightSlopeCut,
			    const std::vector<double>& RightSlopeSmallThreshold, 
			    const std::vector<double>& RightSlopeSmallCut,
                            const std::vector<double>& TS4TS5UpperThreshold,
                            const std::vector<double>& TS4TS5UpperCut,
                            const std::vector<double>& TS4TS5LowerThreshold,
                            const std::vector<double>& TS4TS5LowerCut,
			    bool UseDualFit,
			    bool TriangleIgnoreSlow,
                            bool setLegacyFlags = true);

   ~HBHEPulseShapeFlagSetter();

   void Clear();
   void Initialize();

   template<class Dataframe>
   void SetPulseShapeFlags(HBHERecHit& hbhe, const Dataframe& digi,
                           const HcalCoder& coder, const HcalCalibrations& calib);
private:
   double mMinimumChargeThreshold;
   double mTS4TS5ChargeThreshold;
   double mTS3TS4UpperChargeThreshold;
   double mTS5TS6UpperChargeThreshold;
   double mTS3TS4ChargeThreshold;
   double mTS5TS6ChargeThreshold;
   double mR45PlusOneRange;
   double mR45MinusOneRange;
   int mTrianglePeakTS;
   std::vector<double>  mCharge;  // stores charge for each TS in each digi
   // the pair is defined as (threshold, cut position)
   std::vector<std::pair<double, double> > mLambdaLinearCut;
   std::vector<std::pair<double, double> > mLambdaRMS8MaxCut;
   std::vector<std::pair<double, double> > mLeftSlopeCut;
   std::vector<std::pair<double, double> > mRightSlopeCut;
   std::vector<std::pair<double, double> > mRightSlopeSmallCut;
   std::vector<std::pair<double, double> > mTS4TS5UpperCut;
   std::vector<std::pair<double, double> > mTS4TS5LowerCut;
   bool mUseDualFit;
   bool mTriangleIgnoreSlow;
   bool mSetLegacyFlags;
   std::vector<double> CumulativeIdealPulse;
private:
   TriangleFitResult PerformTriangleFit(const std::vector<double> &Charge);
   double PerformNominalFit(const std::vector<double> &Charge);
   double PerformDualNominalFit(const std::vector<double> &Charge);
   double DualNominalFitSingleTry(const std::vector<double> &Charge, int Offset, int Distance, bool newCharges=true);
   double CalculateRMS8Max(const std::vector<double> &Charge);
   double PerformLinearFit(const std::vector<double> &Charge);
private:
   bool CheckPassFilter(double Charge, double Discriminant, std::vector<std::pair<double, double> > &Cuts,
      int Side);
   std::vector<double> f1_;
   std::vector<double> f2_;
   std::vector<double> errors_;

};
//---------------------------------------------------------------------------
struct TriangleFitResult
{
   double Chi2;
   double LeftSlope;
   double RightSlope;
};
//---------------------------------------------------------------------------

template<class Dataframe>
void HBHEPulseShapeFlagSetter::SetPulseShapeFlags(
    HBHERecHit& hbhe, const Dataframe& digi,
    const HcalCoder& coder, const HcalCalibrations& calib)
{
   //
   // Decide if a digi/pulse is good or bad using fit-based discriminants
   //
   // SetPulseShapeFlags determines the total charge in the digi.
   // If the charge is above the minimum threshold, the code then
   // runs the flag-setting algorithms to determine whether the
   // flags should be set.
   //

   // hack to exclude ieta=28/29 for the moment... 
   int abseta = hbhe.id().ietaAbs();
   if(abseta == 28 || abseta == 29)  return;

   CaloSamples Tool;
   coder.adc2fC(digi, Tool);

   //   mCharge.clear();  // mCharge is a vector of (pedestal-subtracted) Charge values vs. time slice
   const unsigned nRead = digi.size();
   mCharge.resize(nRead);

   double TotalCharge = 0.0;
   for (unsigned i = 0; i < nRead; ++i)
   {
      mCharge[i] = Tool[i] - calib.pedestal(digi[i].capid());
      TotalCharge += mCharge[i];
   }

   // No flagging if TotalCharge is less than threshold
   if(TotalCharge < mMinimumChargeThreshold)
      return;

   if (mSetLegacyFlags)
   {
       double NominalChi2 = 0; 
       if (mUseDualFit == true)
           NominalChi2=PerformDualNominalFit(mCharge); 
       else
           NominalChi2=PerformNominalFit(mCharge);

       double LinearChi2 = PerformLinearFit(mCharge);

       double RMS8Max = CalculateRMS8Max(mCharge);

       // Set the HBHEFlatNoise and HBHESpikeNoise flags
       if(CheckPassFilter(TotalCharge, log(LinearChi2) - log(NominalChi2), mLambdaLinearCut, -1) == false)
           hbhe.setFlagField(1, HcalCaloFlagLabels::HBHEFlatNoise);
       if(CheckPassFilter(TotalCharge, log(RMS8Max) * 2 - log(NominalChi2), mLambdaRMS8MaxCut, -1) == false)
           hbhe.setFlagField(1, HcalCaloFlagLabels::HBHESpikeNoise);

       // Set the HBHETriangleNoise flag
       if ((int)mCharge.size() >= mTrianglePeakTS)  // can't compute flag if peak TS isn't present; revise this at some point?
       {
           TriangleFitResult TriangleResult = PerformTriangleFit(mCharge);

           // initial values
           double TS4Left = 1000;
           double TS4Right = 1000;
 
           // Use 'if' statements to protect against slopes that are either 0 or very small
           if (TriangleResult.LeftSlope > 1e-5)
               TS4Left = mCharge[mTrianglePeakTS] / TriangleResult.LeftSlope;
           if (TriangleResult.RightSlope < -1e-5)
               TS4Right = mCharge[mTrianglePeakTS] / -TriangleResult.RightSlope;
     
           if(TS4Left > 1000 || TS4Left < -1000)
               TS4Left = 1000;
           if(TS4Right > 1000 || TS4Right < -1000)
               TS4Right = 1000;
     
           if(mTriangleIgnoreSlow == false)   // the slow-rising and slow-dropping edges won't be useful in 50ns/75ns
           {
               if(CheckPassFilter(mCharge[mTrianglePeakTS], TS4Left, mLeftSlopeCut, 1) == false)
                   hbhe.setFlagField(1, HcalCaloFlagLabels::HBHETriangleNoise);
               else if(CheckPassFilter(mCharge[mTrianglePeakTS], TS4Right, mRightSlopeCut, 1) == false)
                   hbhe.setFlagField(1, HcalCaloFlagLabels::HBHETriangleNoise);
           }
     
           // fast-dropping ones should be checked in any case
           if(CheckPassFilter(mCharge[mTrianglePeakTS], TS4Right, mRightSlopeSmallCut, -1) == false)
               hbhe.setFlagField(1, HcalCaloFlagLabels::HBHETriangleNoise);
       }
   }

   if(mCharge[4] + mCharge[5] > mTS4TS5ChargeThreshold && mTS4TS5ChargeThreshold>0) // silly protection against negative charge values
   {
      double TS4TS5 = (mCharge[4] - mCharge[5]) / (mCharge[4] + mCharge[5]);
      if(CheckPassFilter(mCharge[4] + mCharge[5], TS4TS5, mTS4TS5UpperCut, 1) == false)
         hbhe.setFlagField(1, HcalCaloFlagLabels::HBHETS4TS5Noise);
      if(CheckPassFilter(mCharge[4] + mCharge[5], TS4TS5, mTS4TS5LowerCut, -1) == false)
         hbhe.setFlagField(1, HcalCaloFlagLabels::HBHETS4TS5Noise);
      
      if(CheckPassFilter(mCharge[4] + mCharge[5], TS4TS5, mTS4TS5UpperCut, 1) == false            && // TS4TS5 is above envelope
         mCharge[3] + mCharge[4] > mTS3TS4ChargeThreshold       &&       mTS3TS4ChargeThreshold>0 && // enough charge in 34
         mCharge[5] + mCharge[6] < mTS5TS6UpperChargeThreshold  &&  mTS5TS6UpperChargeThreshold>0 && // low charge in 56
      	 fabs( (mCharge[4] - mCharge[5]) / (mCharge[4] + mCharge[5]) - 1.0 ) < mR45PlusOneRange    ) // R45 is around +1
   	{
           double TS3TS4 = (mCharge[3] - mCharge[4]) / (mCharge[3] + mCharge[4]);
           if(CheckPassFilter(mCharge[3] + mCharge[4], TS3TS4, mTS4TS5UpperCut,  1) == true && // use the same envelope as TS4TS5
	      CheckPassFilter(mCharge[3] + mCharge[4], TS3TS4, mTS4TS5LowerCut, -1) == true && // use the same envelope as TS4TS5
	      TS3TS4>(mR45MinusOneRange-1)                                                   ) // horizontal cut on R34 (R34>-0.8)
	       hbhe.setFlagField(1, HcalCaloFlagLabels::HBHEOOTPU); // set to 1 if there is a pulse-shape-wise good OOTPU in TS3TS4.
   	}

      if(CheckPassFilter(mCharge[4] + mCharge[5], TS4TS5, mTS4TS5LowerCut, -1) == false            && // TS4TS5 is below envelope
         mCharge[3] + mCharge[4] < mTS3TS4UpperChargeThreshold  &&  mTS3TS4UpperChargeThreshold>0  && // low charge in 34
         mCharge[5] + mCharge[6] > mTS5TS6ChargeThreshold       &&       mTS5TS6ChargeThreshold>0  && // enough charge in 56
         fabs( (mCharge[4] - mCharge[5]) / (mCharge[4] + mCharge[5]) + 1.0 ) < mR45MinusOneRange    ) // R45 is around -1
        {
           double TS5TS6 = (mCharge[5] - mCharge[6]) / (mCharge[5] + mCharge[6]);
           if(CheckPassFilter(mCharge[5] + mCharge[6], TS5TS6, mTS4TS5UpperCut,  1) == true && // use the same envelope as TS4TS5
	      CheckPassFilter(mCharge[5] + mCharge[6], TS5TS6, mTS4TS5LowerCut, -1) == true && // use the same envelope as TS4TS5
	      TS5TS6<(1-mR45PlusOneRange)                                                    ) // horizontal cut on R56 (R56<+0.8)
	       hbhe.setFlagField(1, HcalCaloFlagLabels::HBHEOOTPU); // set to 1 if there is a pulse-shape-wise good OOTPU in TS5TS6.
        }
   }
}

#endif
