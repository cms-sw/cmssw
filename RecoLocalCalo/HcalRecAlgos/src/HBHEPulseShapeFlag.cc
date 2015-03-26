//---------------------------------------------------------------------------
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cmath>

//---------------------------------------------------------------------------
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHEPulseShapeFlag.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"
//---------------------------------------------------------------------------
HBHEPulseShapeFlagSetter::HBHEPulseShapeFlagSetter()
{
   //
   // Argumentless constructor; should not be used
   // 
   // If arguments not properly specified for the constructor, I don't think
   // we'd trust the flagging algorithm.
   // Set the minimum charge threshold large enough so that nothing will be flagged.
   // 

   mMinimumChargeThreshold = 99999999;
   mTS4TS5ChargeThreshold = 99999999;
}
//---------------------------------------------------------------------------
HBHEPulseShapeFlagSetter::HBHEPulseShapeFlagSetter(double MinimumChargeThreshold,
   double TS4TS5ChargeThreshold,
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
   const std::vector<double>& TS4TS5LowerThreshold,
   const std::vector<double>& TS4TS5LowerCut,
   const std::vector<double>& TS4TS5UpperThreshold,
   const std::vector<double>& TS4TS5UpperCut,
   bool UseDualFit, 
   bool TriangleIgnoreSlow)
{
   //
   // The constructor that should be used
   //
   // Copies various thresholds and limits and parameters to the class for future use.
   // Also calls the Initialize() function
   //

   mMinimumChargeThreshold = MinimumChargeThreshold;
   mTS4TS5ChargeThreshold = TS4TS5ChargeThreshold;
   mTrianglePeakTS = TrianglePeakTS;
   mTriangleIgnoreSlow = TriangleIgnoreSlow;

   for(std::vector<double>::size_type i = 0; i < LinearThreshold.size() && i < LinearCut.size(); i++)
      mLambdaLinearCut.push_back(std::pair<double, double>(LinearThreshold[i], LinearCut[i]));
   sort(mLambdaLinearCut.begin(), mLambdaLinearCut.end());

   for(std::vector<double>::size_type i = 0; i < RMS8MaxThreshold.size() && i < RMS8MaxCut.size(); i++)
      mLambdaRMS8MaxCut.push_back(std::pair<double, double>(RMS8MaxThreshold[i], RMS8MaxCut[i]));
   sort(mLambdaRMS8MaxCut.begin(), mLambdaRMS8MaxCut.end());

   for(std::vector<double>::size_type i = 0; i < LeftSlopeThreshold.size() && i < LeftSlopeCut.size(); i++)
      mLeftSlopeCut.push_back(std::pair<double, double>(LeftSlopeThreshold[i], LeftSlopeCut[i]));
   sort(mLeftSlopeCut.begin(), mLeftSlopeCut.end());

   for(std::vector<double>::size_type i = 0; i < RightSlopeThreshold.size() && i < RightSlopeCut.size(); i++)
      mRightSlopeCut.push_back(std::pair<double, double>(RightSlopeThreshold[i], RightSlopeCut[i]));
   sort(mRightSlopeCut.begin(), mRightSlopeCut.end());

   for(std::vector<double>::size_type i = 0; i < RightSlopeSmallThreshold.size() && i < RightSlopeSmallCut.size(); i++)
      mRightSlopeSmallCut.push_back(std::pair<double, double>(RightSlopeSmallThreshold[i], RightSlopeSmallCut[i]));
   sort(mRightSlopeSmallCut.begin(), mRightSlopeSmallCut.end());
   
   for(std::vector<double>::size_type i = 0; i < TS4TS5UpperThreshold.size() && i < TS4TS5UpperCut.size(); i++)
      mTS4TS5UpperCut.push_back(std::pair<double, double>(TS4TS5UpperThreshold[i], TS4TS5UpperCut[i]));
   sort(mTS4TS5UpperCut.begin(), mTS4TS5UpperCut.end());

   for(std::vector<double>::size_type i = 0; i < TS4TS5LowerThreshold.size() && i < TS4TS5LowerCut.size(); i++)
      mTS4TS5LowerCut.push_back(std::pair<double, double>(TS4TS5LowerThreshold[i], TS4TS5LowerCut[i]));
   sort(mTS4TS5LowerCut.begin(), mTS4TS5LowerCut.end());

   mUseDualFit = UseDualFit;

   Initialize();
}
//---------------------------------------------------------------------------
HBHEPulseShapeFlagSetter::~HBHEPulseShapeFlagSetter()
{
   // Dummy destructor - there's nothing to destruct by hand here
}
//---------------------------------------------------------------------------
void HBHEPulseShapeFlagSetter::Clear()
{
   // Dummy function in case something needs to be cleaned....but none right now
}
//---------------------------------------------------------------------------
void HBHEPulseShapeFlagSetter::SetPulseShapeFlags(HBHERecHit &hbhe, 
   const HBHEDataFrame &digi,
   const HcalCoder &coder, 
   const HcalCalibrations &calib)
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
   mCharge.resize(digi.size());

   double TotalCharge = 0;

   for(int i = 0; i < digi.size(); ++i)
   {
      mCharge[i] = Tool[i] - calib.pedestal(digi.sample(i).capid());
      TotalCharge += mCharge[i];
   }

   // No flagging if TotalCharge is less than threshold
   if(TotalCharge < mMinimumChargeThreshold)
      return;

   double NominalChi2 = 0; 
   if (mUseDualFit == true)
     NominalChi2=PerformDualNominalFit(mCharge); 
   else
     NominalChi2=PerformNominalFit(mCharge);

   double LinearChi2 = PerformLinearFit(mCharge);

   double RMS8Max = CalculateRMS8Max(mCharge);
   TriangleFitResult TriangleResult = PerformTriangleFit(mCharge);

   // Set the HBHEFlatNoise and HBHESpikeNoise flags
   if(CheckPassFilter(TotalCharge, log(LinearChi2) - log(NominalChi2), mLambdaLinearCut, -1) == false)
      hbhe.setFlagField(1, HcalCaloFlagLabels::HBHEFlatNoise);
   if(CheckPassFilter(TotalCharge, log(RMS8Max) * 2 - log(NominalChi2), mLambdaRMS8MaxCut, -1) == false)
      hbhe.setFlagField(1, HcalCaloFlagLabels::HBHESpikeNoise);

   // Set the HBHETriangleNoise flag
   if ((int)mCharge.size() >= mTrianglePeakTS)  // can't compute flag if peak TS isn't present; revise this at some point?
   {
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

   if(mCharge[4] + mCharge[5] > mTS4TS5ChargeThreshold && mTS4TS5ChargeThreshold>0) // silly protection against negative charge values
   {
      double TS4TS5 = (mCharge[4] - mCharge[5]) / (mCharge[4] + mCharge[5]);
      if(CheckPassFilter(mCharge[4] + mCharge[5], TS4TS5, mTS4TS5UpperCut, 1) == false)
         hbhe.setFlagField(1, HcalCaloFlagLabels::HBHETS4TS5Noise);
      if(CheckPassFilter(mCharge[4] + mCharge[5], TS4TS5, mTS4TS5LowerCut, -1) == false)
         hbhe.setFlagField(1, HcalCaloFlagLabels::HBHETS4TS5Noise);
   }
}
//---------------------------------------------------------------------------
void HBHEPulseShapeFlagSetter::Initialize()
{
   // 
   // Initialization: whatever preprocess is needed
   //
   // 1. Get the ideal pulse shape from CMSSW
   //
   //    Since the HcalPulseShapes class stores the ideal pulse shape in terms of 256 numbers,
   //    each representing 1ns integral of the ideal pulse, here I'm taking out the vector
   //    by calling at() function.
   //
   //    A cumulative distribution is formed and stored to save some time doing integration to TS later on
   //
   // 2. Reserve space for vector
   //

   std::vector<double> PulseShape;

   HcalPulseShapes Shapes;
   HcalPulseShapes::Shape HPDShape = Shapes.hbShape();

   PulseShape.reserve(350);
   for(int i = 0; i < 200; i++)
      PulseShape.push_back(HPDShape.at(i));
   PulseShape.insert(PulseShape.begin(), 150, 0);   // Safety margin of a lot of zeros in the beginning

   CumulativeIdealPulse.reserve(350);
   CumulativeIdealPulse.clear();
   CumulativeIdealPulse.push_back(0);
   for(unsigned int i = 1; i < PulseShape.size(); i++)
      CumulativeIdealPulse.push_back(CumulativeIdealPulse[i-1] + PulseShape[i]);

   // reserve space for vector
   mCharge.reserve(10);
}
//---------------------------------------------------------------------------
TriangleFitResult HBHEPulseShapeFlagSetter::PerformTriangleFit(const std::vector<double> &Charge)
{
   //
   // Perform a "triangle fit", and extract the slopes
   //
   // Left-hand side and right-hand side are not correlated to each other - do them separately
   //

   TriangleFitResult result;
   result.Chi2 = 0;
   result.LeftSlope = 0;
   result.RightSlope = 0;

   int DigiSize = Charge.size();

   // right side, starting from TS4
   double MinimumRightChi2 = 1000000;
   double Numerator = 0;
   double Denominator = 0;

   for(int iTS = mTrianglePeakTS + 2; iTS <= DigiSize; iTS++)   // the place where first TS center in flat line
   {
      // fit a straight line to the triangle part
      Numerator = 0;
      Denominator = 0;

      for(int i = mTrianglePeakTS + 1; i < iTS; i++)
      {
         Numerator += (i - mTrianglePeakTS) * (Charge[i] - Charge[mTrianglePeakTS]);
         Denominator += (i - mTrianglePeakTS) * (i - mTrianglePeakTS);
      }

      double BestSlope = 0;
      if (Denominator!=0) BestSlope = Numerator / Denominator;
      if(BestSlope > 0)
         BestSlope = 0;

      // check if the slope is reasonable
      if(iTS != DigiSize)
	{
	  // iTS starts at mTrianglePeak+2; denominator never zero
	  if(BestSlope > -1 * Charge[mTrianglePeakTS] / (iTS - mTrianglePeakTS))
            BestSlope = -1 * Charge[mTrianglePeakTS] / (iTS - mTrianglePeakTS);
	  if(BestSlope < -1 * Charge[mTrianglePeakTS] / (iTS - 1 - mTrianglePeakTS))
	    BestSlope = -1 * Charge[mTrianglePeakTS] / (iTS - 1 - mTrianglePeakTS);
	}
      else
	{
	  // iTS starts at mTrianglePeak+2; denominator never zero
	  if(BestSlope < -1 * Charge[mTrianglePeakTS] / (iTS - 1 - mTrianglePeakTS)) 
            BestSlope = -1 * Charge[mTrianglePeakTS] / (iTS - 1 - mTrianglePeakTS);
	}
      
      // calculate partial chi2

      // The shape I'm fitting is more like a tent than a triangle.
      // After the end of triangle edge assume a flat line

      double Chi2 = 0;
      for(int i = mTrianglePeakTS + 1; i < iTS; i++)
         Chi2 += (Charge[mTrianglePeakTS] - Charge[i] + (i - mTrianglePeakTS) * BestSlope)
            * (Charge[mTrianglePeakTS] - Charge[i] + (i - mTrianglePeakTS) * BestSlope);
      for(int i = iTS; i < DigiSize; i++)    // Assumes fit line = 0 for iTS > fit
         Chi2 += Charge[i] * Charge[i];

      if(Chi2 < MinimumRightChi2)
      {
         MinimumRightChi2 = Chi2;
         result.RightSlope = BestSlope;
      }
   }   // end of right-hand side loop

   // left side
   double MinimumLeftChi2 = 1000000;

   for(int iTS = 0; iTS < (int)mTrianglePeakTS; iTS++)   // the first time after linear fit ends
   {
      // fit a straight line to the triangle part
      Numerator = 0;
      Denominator = 0;
      for(int i = iTS; i < (int)mTrianglePeakTS; i++)
      {
         Numerator = Numerator + (i - mTrianglePeakTS) * (Charge[i] - Charge[mTrianglePeakTS]);
         Denominator = Denominator + (i - mTrianglePeakTS) * (i - mTrianglePeakTS);
      }

      double BestSlope = 0;
      if (Denominator!=0) BestSlope = Numerator / Denominator;
      if (BestSlope < 0)
	BestSlope = 0;

      // check slope
      if(iTS != 0)
	{
	  // iTS must be >0 and < mTrianglePeakTS; slope never 0
	  if(BestSlope > Charge[mTrianglePeakTS] / (mTrianglePeakTS - iTS))
            BestSlope = Charge[mTrianglePeakTS] / (mTrianglePeakTS - iTS);
	  if(BestSlope < Charge[mTrianglePeakTS] / (mTrianglePeakTS + 1 - iTS))
            BestSlope = Charge[mTrianglePeakTS] / (mTrianglePeakTS + 1 - iTS);
	}
      else
	{
	  if(BestSlope > Charge[mTrianglePeakTS] / (mTrianglePeakTS - iTS))
            BestSlope = Charge[mTrianglePeakTS] / (mTrianglePeakTS - iTS);
	}
      
      // calculate minimum chi2
      double Chi2 = 0;
      for(int i = 0; i < iTS; i++)
         Chi2 += Charge[i] * Charge[i];
      for(int i = iTS; i < (int)mTrianglePeakTS; i++)
         Chi2 += (Charge[mTrianglePeakTS] - Charge[i] + (i - mTrianglePeakTS) * BestSlope)
            * (Charge[mTrianglePeakTS] - Charge[i] + (i - mTrianglePeakTS) * BestSlope);

      if(MinimumLeftChi2 > Chi2)
      {
         MinimumLeftChi2 = Chi2;
         result.LeftSlope = BestSlope;
      }
   }   // end of left-hand side loop

   result.Chi2 = MinimumLeftChi2 + MinimumRightChi2;

   return result;
}
//---------------------------------------------------------------------------
double HBHEPulseShapeFlagSetter::PerformNominalFit(const std::vector<double> &Charge)
{
   //
   // Performs a fit to the ideal pulse shape.  Returns best chi2
   //
   // A scan over different timing offset (for the ideal pulse) is carried out,
   //    and for each offset setting a one-parameter fit is performed
   //

   int DigiSize = Charge.size();

   double MinimumChi2 = 100000;

   double F = 0;

   double SumF2 = 0;
   double SumTF = 0;
   double SumT2 = 0;

   for(int i = 0; i + 250 < (int)CumulativeIdealPulse.size(); i++)
   {
      if(CumulativeIdealPulse[i+250] - CumulativeIdealPulse[i] < 1e-5)
         continue;

      SumF2 = 0;
      SumTF = 0;
      SumT2 = 0;

      double ErrorTemp=0;
      for(int j = 0; j < DigiSize; j++)
	{
	  // get ideal pulse component for this time slice....
	  F = CumulativeIdealPulse[i+j*25+25] - CumulativeIdealPulse[i+j*25];
	  
	  ErrorTemp=Charge[j];
	  if (ErrorTemp<1) // protection against small charges
	    ErrorTemp=1;
	  // ...and increment various summations
	  SumF2 += F * F / ErrorTemp;
	  SumTF += F * Charge[j] / ErrorTemp;
	  SumT2 += fabs(Charge[j]);
	}
      
      /* 
	 chi2= sum((Charge[j]-aF)^2/|Charge[j]|
	 ( |Charge[j]| = assumed sigma^2 for Charge[j]; a bit wonky for Charge[j]<1 )
         chi2 = sum(|Charge[j]|) - 2*sum(aF*Charge[j]/|Charge[j]|) +sum( a^2*F^2/|Charge[j]|)
	 chi2 minimimized when d(chi2)/da = 0:
         a = sum(F*Charge[j])/sum(F^2)
	 ...
         chi2= sum(|Q[j]|) - sum(Q[j]/|Q[j]|*F)*sum(Q[j]/|Q[j]|*F)/sum(F^2/|Q[j]|), where Q = Charge
	 chi2 = SumT2 - SumTF*SumTF/SumF2
      */
      
      double Chi2 = SumT2 - SumTF * SumTF / SumF2;
      
      if(Chi2 < MinimumChi2)
	MinimumChi2 = Chi2;
   }
   
   // safety protection in case of perfect fit - don't want the log(...) to explode
   if(MinimumChi2 < 1e-5)
      MinimumChi2 = 1e-5;

   return MinimumChi2;
}
//---------------------------------------------------------------------------
double HBHEPulseShapeFlagSetter::PerformDualNominalFit(const std::vector<double> &Charge)
{
   //
   // Perform dual nominal fit and returns the chi2
   // 
   // In this function we do a scan over possible "distance" (number of time slices between two components)
   //    and overall offset for the two components; first coarse, then finer
   // All the fitting is done in the DualNominalFitSingleTry function
   //

   double OverallMinimumChi2 = 1000000;

   int AvailableDistance[] = {-100, -75, -50, 50, 75, 100};

   // loop over possible pulse distances between two components
   bool isFirst=true;

   for(int k = 0; k < 6; k++)
   {
      double SingleMinimumChi2 = 1000000;
      int MinOffset = 0;

      // scan coarsely through different offsets and find the minimum
      for(int i = 0; i + 250 < (int)CumulativeIdealPulse.size(); i += 10)
      {
	double Chi2 = DualNominalFitSingleTry(Charge, i, AvailableDistance[k],isFirst);
	isFirst=false;
         if(Chi2 < SingleMinimumChi2)
         {
            SingleMinimumChi2 = Chi2;
            MinOffset = i;
         }
      }

      // around the minimum, scan finer for better a better minimum
      for(int i = MinOffset - 15; i + 250 < (int)CumulativeIdealPulse.size() && i < MinOffset + 15; i++)
      {
	double Chi2 = DualNominalFitSingleTry(Charge, i, AvailableDistance[k],false);
         if(Chi2 < SingleMinimumChi2)
            SingleMinimumChi2 = Chi2;
      }

      // update overall minimum chi2
      if(SingleMinimumChi2 < OverallMinimumChi2)
         OverallMinimumChi2 = SingleMinimumChi2;
   }

   return OverallMinimumChi2;
}
//---------------------------------------------------------------------------
double HBHEPulseShapeFlagSetter::DualNominalFitSingleTry(const std::vector<double> &Charge, int Offset, int Distance, bool newCharges)
{
   //
   // Does a fit to dual signal pulse hypothesis given offset and distance of the two target pulses
   //
   // The only parameters to fit here are the two pulse heights of in-time and out-of-time components
   //    since offset is given
   // The calculation here is based from writing down the Chi2 formula and minimize against the two parameters,
   //    ie., Chi2 = Sum{((T[i] - a1 * F1[i] - a2 * F2[i]) / (Sigma[i]))^2}, where T[i] is the input pulse shape,
   //    and F1[i], F2[i] are the two ideal pulse components
   //

   int DigiSize = Charge.size();

   if(Offset < 0 || Offset + 250 >= (int)CumulativeIdealPulse.size())
      return 1000000;
   if(CumulativeIdealPulse[Offset+250] - CumulativeIdealPulse[Offset] < 1e-5)
      return 1000000;

   if ( newCharges) {
     f1_.resize(DigiSize);
     f2_.resize(DigiSize);
     errors_.resize(DigiSize);
     for(int j = 0; j < DigiSize; j++)
       {
	 errors_[j] = Charge[j];
	 if(errors_[j] < 1)
	   errors_[j] = 1;
	 errors_[j]=1.0/errors_[j];
       }
   }

   double SumF1F1 = 0;
   double SumF1F2 = 0;
   double SumF2F2 = 0;
   double SumTF1 = 0;
   double SumTF2 = 0;

   unsigned int cipSize=CumulativeIdealPulse.size();
   for(int j = 0; j < DigiSize; j++)
   {
      // this is the TS value for in-time component - no problem we can do a subtraction directly
      f1_[j] = CumulativeIdealPulse[Offset+j*25+25] - CumulativeIdealPulse[Offset+j*25];

      // However for the out-of-time component the index might go out-of-bound.
      // Let's protect against this.

      int OffsetTemp = Offset + j * 25 + Distance;
      
      double C1 = 0;   // lower-indexed value in the cumulative pulse shape
      double C2 = 0;   // higher-indexed value in the cumulative pulse shape

      
      if(OffsetTemp + 25 >= (int)cipSize)
	C1 = CumulativeIdealPulse[cipSize-1];
      else
	if( OffsetTemp  >= -25)
	  C1 = CumulativeIdealPulse[OffsetTemp+25];
      if(OffsetTemp >= (int)cipSize)
	C2 = CumulativeIdealPulse[cipSize-1];
      else
	if( OffsetTemp >= 0)
	  C2 = CumulativeIdealPulse[OffsetTemp];
      f2_[j] = C1 - C2;

      SumF1F1 += f1_[j] * f1_[j] * errors_[j];
      SumF1F2 += f1_[j] * f2_[j] * errors_[j]; 
      SumF2F2 += f2_[j] * f2_[j] * errors_[j];
      SumTF1  += f1_[j] * Charge[j] * errors_[j]; 
      SumTF2  += f2_[j] * Charge[j] * errors_[j]; 
   }

   double Height  = 0;
   double Height2 = 0;
     if (fabs(SumF1F2*SumF1F2-SumF1F1*SumF2F2)>1e-5)
       {
	 Height  = (SumF1F2 * SumTF2 - SumF2F2 * SumTF1) / (SumF1F2 * SumF1F2 - SumF1F1 * SumF2F2);
	 Height2 = (SumF1F2 * SumTF1 - SumF1F1 * SumTF2) / (SumF1F2 * SumF1F2 - SumF1F1 * SumF2F2);
       }

   double Chi2 = 0;
   for(int j = 0; j < DigiSize; j++)
   {
      double Residual = Height * f1_[j] + Height2 * f2_[j] - Charge[j];  
      Chi2 += Residual * Residual *errors_[j];                             
   } 

   // Safety protection in case of zero
   if(Chi2 < 1e-5)
      Chi2 = 1e-5;

   return Chi2;
}
//---------------------------------------------------------------------------
double HBHEPulseShapeFlagSetter::CalculateRMS8Max(const std::vector<double> &Charge)
{
   //
   // CalculateRMS8Max
   //
   // returns "RMS" divided by the largest charge in the time slices
   //    "RMS" is calculated using all but the two largest time slices.
   //    The "RMS" is not quite the actual RMS (see note below), but the value is only
   //    used for determining max values, and is not quoted as the actual RMS anywhere.
   //

   int DigiSize = Charge.size();

   if (DigiSize<=2)  return 1e-5;  // default statement when DigiSize is too small for useful RMS calculation
  // Copy Charge vector again, since we are passing references around
   std::vector<double> TempCharge = Charge;

   // Sort TempCharge vector from smallest to largest charge
   sort(TempCharge.begin(), TempCharge.end());

   double Total = 0;
   double Total2 = 0;
   for(int i = 0; i < DigiSize - 2; i++)
   {
      Total = Total + TempCharge[i];
      Total2 = Total2 + TempCharge[i] * TempCharge[i];
   }

   // This isn't quite the RMS (both Total2 and Total*Total need to be
   // divided by an extra (DigiSize-2) within the sqrt to get the RMS.)
   // We're only using this value for relative comparisons, though; we
   // aren't explicitly interpreting it as the RMS.  It might be nice
   // to either change the calculation or rename the variable in the future, though.

   double RMS = sqrt(Total2 - Total * Total / (DigiSize - 2));

   double RMS8Max = RMS / TempCharge[DigiSize-1];
   if(RMS8Max < 1e-5)   // protection against zero
      RMS8Max = 1e-5;

   return RMS / TempCharge[DigiSize-1];
}
//---------------------------------------------------------------------------
double HBHEPulseShapeFlagSetter::PerformLinearFit(const std::vector<double> &Charge)
{
   //
   // Performs a straight-line fit over all time slices, and returns the chi2 value
   //
   // The calculation here is based from writing down the formula for chi2 and minimize
   //    with respect to the parameters in the fit, ie., slope and intercept of the straight line
   // By doing two differentiation, we will get two equations, and the best parameters are determined by these
   //

   int DigiSize = Charge.size();

   double SumTS2OverTi = 0;
   double SumTSOverTi = 0;
   double SumOverTi = 0;
   double SumTiTS = 0;
   double SumTi = 0;

   double Error2 = 0;
   for(int i = 0; i < DigiSize; i++)
   {
      Error2 = Charge[i];
      if(Charge[i] < 1)
         Error2 = 1;

      SumTS2OverTi += 1.* i * i / Error2;
      SumTSOverTi  += 1.* i / Error2;
      SumOverTi    += 1. / Error2;
      SumTiTS      += Charge[i] * i / Error2;
      SumTi        += Charge[i] / Error2;
   }

   double CM1 = SumTS2OverTi;   // Coefficient in front of slope in equation 1
   double CM2 = SumTSOverTi;   // Coefficient in front of slope in equation 2
   double CD1 = SumTSOverTi;   // Coefficient in front of intercept in equation 1
   double CD2 = SumOverTi;   // Coefficient in front of intercept in equation 2
   double C1 = SumTiTS;   // Constant coefficient in equation 1
   double C2 = SumTi;   // Constant coefficient in equation 2

   // Denominators always non-zero by construction
   double Slope = (C1 * CD2 - C2 * CD1) / (CM1 * CD2 - CM2 * CD1);
   double Intercept = (C1 * CM2 - C2 * CM1) / (CD1 * CM2 - CD2 * CM1);

   // now that the best parameters are found, calculate chi2 from those
   double Chi2 = 0;
   for(int i = 0; i < DigiSize; i++)
   {
      double Deviation = Slope * i + Intercept - Charge[i];
      double Error2 = Charge[i];
      if(Charge[i] < 1)
         Error2 = 1;
      Chi2 += Deviation * Deviation / Error2;  
   }

   // safety protection in case of perfect fit
   if(Chi2 < 1e-5)
      Chi2 = 1e-5;

   return Chi2;
}
//---------------------------------------------------------------------------
bool HBHEPulseShapeFlagSetter::CheckPassFilter(double Charge,
					       double Discriminant,
					       std::vector<std::pair<double, double> > &Cuts, 
					       int Side)
{
   //
   // Checks whether Discriminant value passes Cuts for the specified Charge.  True if pulse is good.
   //
   // The "Cuts" pairs are assumed to be sorted in terms of size from small to large,
   //    where each "pair" = (Charge, Discriminant)
   // "Side" is either positive or negative, which determines whether to discard the pulse if discriminant
   //    is greater or smaller than the cut value
   //

   if(Cuts.size() == 0)   // safety check that there are some cuts defined
      return true;

   if(Charge <= Cuts[0].first)   // too small to cut on
      return true;

   int IndexLargerThanCharge = -1;   // find the range it is falling in
   for(int i = 1; i < (int)Cuts.size(); i++)
   {
      if(Cuts[i].first > Charge)
      {
         IndexLargerThanCharge = i;
         break;
      }
   }

   double limit = 1000000;

   if(IndexLargerThanCharge == -1)   // if charge is greater than the last entry, assume flat line
      limit = Cuts[Cuts.size()-1].second;
   else   // otherwise, do a linear interpolation to find the cut position
   {
      double C1 = Cuts[IndexLargerThanCharge].first;
      double C2 = Cuts[IndexLargerThanCharge-1].first;
      double L1 = Cuts[IndexLargerThanCharge].second;
      double L2 = Cuts[IndexLargerThanCharge-1].second;

      limit = (Charge - C1) / (C2 - C1) * (L2 - L1) + L1;
   }

   if(Side > 0 && Discriminant > limit)
      return false;
   if(Side < 0 && Discriminant < limit)
      return false;

   return true;
}
//---------------------------------------------------------------------------


