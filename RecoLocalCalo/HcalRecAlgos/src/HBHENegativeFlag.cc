//---------------------------------------------------------------------------
#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <cmath>

//---------------------------------------------------------------------------
#include "RecoLocalCalo/HcalRecAlgos/interface/HBHENegativeFlag.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"

#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"

#include "CondFormats/DataRecord/interface/HcalOOTPileupCorrectionRcd.h"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionColl.h"
#include "Geometry/CaloTopology/interface/HcalTopology.h"
#include "Geometry/Records/interface/IdealGeometryRecord.h"
//---------------------------------------------------------------------------
HBHENegativeFlagSetter::HBHENegativeFlagSetter()
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
   mFirst = 4;
   mLast = 6;
   mBunchCrossingInfo = NULL;
   mLengthBunchCrossingInfo = 0;
}
//---------------------------------------------------------------------------
HBHENegativeFlagSetter::HBHENegativeFlagSetter(double MinimumChargeThreshold,
   double TS4TS5ChargeThreshold, int First, int Last,
   std::vector<double> threshold, std::vector<double> cut)
{
   //
   // The constructor that should be used
   //
   // Copies various thresholds and limits and parameters to the class for future use.
   // Also calls the Initialize() function
   //

   mMinimumChargeThreshold = MinimumChargeThreshold;
   mTS4TS5ChargeThreshold = TS4TS5ChargeThreshold;
   mFirst = First;
   mLast = Last;

   if(mLast < mFirst)   // sanity protection
      std::swap(mFirst, mLast);

   mBunchCrossingInfo = NULL;
   mLengthBunchCrossingInfo = 0;

   for(int i = 0; i < (int)std::min(threshold.size(), cut.size()); i++)
      mCut.push_back(std::pair<double, double>(threshold[i], cut[i]));
   std::sort(mCut.begin(), mCut.end());

   Initialize();
}
//---------------------------------------------------------------------------
HBHENegativeFlagSetter::~HBHENegativeFlagSetter()
{
   // Dummy destructor - there's nothing to destruct by hand here
}
//---------------------------------------------------------------------------
void HBHENegativeFlagSetter::Clear()
{
   // Dummy function in case something needs to be cleaned....but none right now
}
//---------------------------------------------------------------------------
void HBHENegativeFlagSetter::SetPulseShapeFlags(HBHERecHit &hbhe, 
   const HBHEDataFrame &digi,
   const HcalCoder &coder, 
   const HcalCalibrations &calib)
{
   //
   // Decide if a digi/pulse is good or bad using negative energy discriminants
   //
   // SetPulseShapeFlags determines the total charge in the digi.
   // If the charge is above the minimum threshold, the code then
   // runs the flag-setting algorithms to determine whether the
   // flags should be set.
   //

   if(hbhePileupCorr_)
   {
      CaloSamples cs;
      coder.adc2fC(digi,cs);
      const int nRead = cs.size();

      double inputCharge[CaloSamples::MAXSAMPLES];
      double gains[CaloSamples::MAXSAMPLES];
      double CorrectedEnergy[CaloSamples::MAXSAMPLES];

      for(int i = 0; i < nRead; i++)
      {
         const int capid = digi[i].capid();
         inputCharge[i] = cs[i] - calib.pedestal(capid);
         gains[i] = calib.respcorrgain(capid);
      }

      double ChargeInWindow = 0;
      for(int i = mFirst; i <= mLast && i < CaloSamples::MAXSAMPLES; i++)
         ChargeInWindow = ChargeInWindow + inputCharge[i];
      if(ChargeInWindow < mMinimumChargeThreshold)
         return;

      const bool UseGain = hbhePileupCorr_->inputIsEnergy();
      if(UseGain == true)
         for(int i = 0; i < nRead; i++)
            inputCharge[i] = inputCharge[i] * gains[i];

      bool pulseShapeCorrApplied = false;
      bool leakCorrApplied = false;
      bool readjustTiming = false;

      int n = std::min(mLast + 1, CaloSamples::MAXSAMPLES) - mFirst;

      hbhePileupCorr_->apply(digi.id(), inputCharge, nRead,
            mBunchCrossingInfo, mLengthBunchCrossingInfo, mFirst, n,
            CorrectedEnergy, CaloSamples::MAXSAMPLES,
            &pulseShapeCorrApplied, &leakCorrApplied,
            &readjustTiming);

      for(int i = mFirst; i <= mLast; i++)
      {
         bool Decision = CheckPassFilter(ChargeInWindow, CorrectedEnergy[i], mCut, -1);
         if(Decision == false)
            hbhe.setFlagField(1, HcalCaloFlagLabels::HBHENegativeNoise);
      }
   }
}
//---------------------------------------------------------------------------
void HBHENegativeFlagSetter::Initialize()
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
}
//---------------------------------------------------------------------------
void HBHENegativeFlagSetter::SetHBHEPileupCorrection(boost::shared_ptr<AbsOOTPileupCorrection> corr)
{
    hbhePileupCorr_ = corr;
}
//---------------------------------------------------------------------------
void HBHENegativeFlagSetter::SetBXInfo(const BunchXParameter *info, unsigned length)
{
   mBunchCrossingInfo = info;
   mLengthBunchCrossingInfo = length;
}
//---------------------------------------------------------------------------
bool HBHENegativeFlagSetter::CheckPassFilter(double Charge,
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


