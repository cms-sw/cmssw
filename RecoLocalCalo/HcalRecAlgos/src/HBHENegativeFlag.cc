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

#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"

#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseShapes.h"

#include "CondFormats/DataRecord/interface/HcalOOTPileupCorrectionRcd.h"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrectionColl.h"
#include "CondFormats/HcalObjects/interface/OOTPileupCorrData.h"

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
HBHENegativeFlagSetter::HBHENegativeFlagSetter(double minimumChargeThreshold,
   double tS4TS5ChargeThreshold, int first, int last,
   std::vector<double> threshold, std::vector<double> cut)
{
   //
   // The constructor that should be used
   //
   // Copies various thresholds and limits and parameters to the class for future use.
   //

   mMinimumChargeThreshold = minimumChargeThreshold;
   mTS4TS5ChargeThreshold = tS4TS5ChargeThreshold;
   mFirst = first;
   mLast = last;

   if (mFirst < 2 || mLast < mFirst)
       throw cms::Exception("Invalid mFirst, mLast specification");

   mBunchCrossingInfo = NULL;
   mLengthBunchCrossingInfo = 0;

   for(int i = 0; i < (int)std::min(threshold.size(), cut.size()); i++)
      mCut.push_back(std::pair<double, double>(threshold[i], cut[i]));
   std::sort(mCut.begin(), mCut.end());
}
//---------------------------------------------------------------------------
HBHENegativeFlagSetter::~HBHENegativeFlagSetter()
{
   // Dummy destructor - there's nothing to destruct by hand here
}
//---------------------------------------------------------------------------
void HBHENegativeFlagSetter::setPulseShapeFlags(HBHERecHit &hbhe, 
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

   const OOTPileupCorrData* corrObj = dynamic_cast<OOTPileupCorrData*>(hbhePileupCorr_.get());
   if (corrObj)
   {
      CaloSamples cs;
      coder.adc2fC(digi,cs);
      const int nRead = cs.size();

      double ts[CaloSamples::MAXSAMPLES];
      for (int i=0; i < nRead; i++)
      {
         const int capid = digi[i].capid();
         ts[i] = cs[i] - calib.pedestal(capid);
      }

      double ChargeInWindow = 0.0;
      for(int i = mFirst; i <= mLast && i < CaloSamples::MAXSAMPLES; i++)
         ChargeInWindow += ts[i];
      if(ChargeInWindow < mMinimumChargeThreshold)
         return;

      const OOTPileupCorrDataFcn& fcn = corrObj->getCorrectionByID(hbhe.id());
      const PiecewiseScalingPolynomial& a1 = fcn.getA1();
      const PiecewiseScalingPolynomial& a2 = fcn.getA2();

      bool passes = true;
      for (int i = mFirst; i <= mLast && passes; i++)
      {
          const double ecorr = ts[i] - a1(ts[i-1]) - a2(ts[i-2]);
          passes = checkPassFilter(ChargeInWindow, ecorr, mCut, -1);
      }
      if (!passes)
          hbhe.setFlagField(1, HcalCaloFlagLabels::HBHENegativeNoise);
   }
}
//---------------------------------------------------------------------------
void HBHENegativeFlagSetter::setHBHEPileupCorrection(boost::shared_ptr<AbsOOTPileupCorrection> corr)
{
    hbhePileupCorr_ = corr;
}
//---------------------------------------------------------------------------
void HBHENegativeFlagSetter::setBXInfo(const BunchXParameter *info, unsigned length)
{
   mBunchCrossingInfo = info;
   mLengthBunchCrossingInfo = length;
}
//---------------------------------------------------------------------------
bool HBHENegativeFlagSetter::checkPassFilter(double charge,
					       double discriminant,
					       std::vector<std::pair<double, double> > &cuts, 
					       int side)
{
   //
   // Checks whether Discriminant value passes Cuts for the specified Charge.  True if pulse is good.
   //
   // The "Cuts" pairs are assumed to be sorted in terms of size from small to large,
   //    where each "pair" = (Charge, Discriminant)
   // "Side" is either positive or negative, which determines whether to discard the pulse if discriminant
   //    is greater or smaller than the cut value
   //

   if(cuts.size() == 0)   // safety check that there are some cuts defined
      return true;

   if(charge <= cuts[0].first)   // too small to cut on
      return true;

   int indexLargerThanCharge = -1;   // find the range it is falling in
   for(int i = 1; i < (int)cuts.size(); i++)
   {
      if(cuts[i].first > charge)
      {
         indexLargerThanCharge = i;
         break;
      }
   }

   double limit = 1000000;

   if(indexLargerThanCharge == -1)   // if charge is greater than the last entry, assume flat line
      limit = cuts[cuts.size()-1].second;
   else   // otherwise, do a linear interpolation to find the cut position
   {
      double C1 = cuts[indexLargerThanCharge].first;
      double C2 = cuts[indexLargerThanCharge-1].first;
      double L1 = cuts[indexLargerThanCharge].second;
      double L2 = cuts[indexLargerThanCharge-1].second;

      limit = (charge - C1) / (C2 - C1) * (L2 - L1) + L1;
   }

   if(side > 0 && discriminant > limit)
      return false;
   if(side < 0 && discriminant < limit)
      return false;

   return true;
}
//---------------------------------------------------------------------------


