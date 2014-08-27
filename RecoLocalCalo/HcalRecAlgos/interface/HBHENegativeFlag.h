//---------------------------------------------------------------------------
#ifndef HBHE_NEGATIVE_FLAG_H_IKAJHGEWRHIGKHAWFIKGHAWIKGH
#define HBHE_NEGATIVE_FLAG_H_IKAJHGEWRHIGKHAWFIKGHAWIKGH
//---------------------------------------------------------------------------
// Negative filter algorithms for HBHE noise flagging
// 
// Original Author: Yi Chen (Caltech), (1)3364 (Aug. 21, 2014)
//---------------------------------------------------------------------------
#include <string>
#include <vector>
#include <map>
#include "boost/shared_ptr.hpp"
//---------------------------------------------------------------------------
#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
#include "CondFormats/HcalObjects/interface/AbsOOTPileupCorrection.h"
//---------------------------------------------------------------------------
class HBHENegativeFlagSetter;
//---------------------------------------------------------------------------
class HBHENegativeFlagSetter
{
   public:
      HBHENegativeFlagSetter();
      HBHENegativeFlagSetter(double MinimumChargeThreshold,
            double TS4TS5ChargeThreshold,
            int First, int Last,
            std::vector<double> threshold,
            std::vector<double> cut);
      ~HBHENegativeFlagSetter();
      void Clear();
      void SetPulseShapeFlags(HBHERecHit& hbhe, const HBHEDataFrame &digi,
            const HcalCoder &coder, const HcalCalibrations &calib);
      void Initialize();
      void SetHBHEPileupCorrection(boost::shared_ptr<AbsOOTPileupCorrection> corr);
      void SetBXInfo(const BunchXParameter *info, unsigned length);
   private:
      double mMinimumChargeThreshold;
      double mTS4TS5ChargeThreshold;
      boost::shared_ptr<AbsOOTPileupCorrection> hbhePileupCorr_;
      int mFirst;
      int mLast;
      const BunchXParameter *mBunchCrossingInfo;
      unsigned mLengthBunchCrossingInfo;
      std::vector<std::pair<double, double> > mCut;
   private:
      bool CheckPassFilter(double Charge, double Discriminant, std::vector<std::pair<double, double> > &Cuts,
         int Side);
};
//---------------------------------------------------------------------------
#endif

