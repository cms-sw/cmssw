//---------------------------------------------------------------------------
#ifndef HBHENegativeFlag_H
#define HBHENegativeFlag_H
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
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
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
      HBHENegativeFlagSetter(double minimumChargeThreshold,
            double tS4TS5ChargeThreshold,
            int first, int last,
            std::vector<double> threshold,
            std::vector<double> cut);
      ~HBHENegativeFlagSetter();
      void setPulseShapeFlags(HBHERecHit& hbhe, const HBHEDataFrame &digi,
            const HcalCoder &coder, const HcalCalibrations &calib);
      void setHBHEPileupCorrection(boost::shared_ptr<AbsOOTPileupCorrection> corr);
      void setBXInfo(const BunchXParameter *info, unsigned length);
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
      bool checkPassFilter(double charge, double discriminant, std::vector<std::pair<double, double> > &cuts,
         int side);
};
//---------------------------------------------------------------------------
#endif

