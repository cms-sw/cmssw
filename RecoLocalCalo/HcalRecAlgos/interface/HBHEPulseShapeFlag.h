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
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"
//---------------------------------------------------------------------------
class HBHEPulseShapeFlagSetter;
struct TriangleFitResult;
//---------------------------------------------------------------------------
class HBHEPulseShapeFlagSetter
{
public:
   HBHEPulseShapeFlagSetter();
   HBHEPulseShapeFlagSetter(double MinimumChargeThreshold, 
			    unsigned int TrianglePeakTS,
			    std::vector<double> LinearThreshold, 
			    std::vector<double> LinearCut,
			    std::vector<double> RMS8MaxThreshold,
			    std::vector<double> RMS8MaxCut,
			    std::vector<double> LeftSlopeThreshold, 
			    std::vector<double> LeftSlopeCut,
			    std::vector<double> RightSlopeThreshold, 
			    std::vector<double> RightSlopeCut,
			    std::vector<double> RightSlopeSmallThreshold, 
			    std::vector<double> RightSlopeSmallCut,
			    bool UseDualFit, 
			    bool TriangleIgnoreSlow);
   ~HBHEPulseShapeFlagSetter();
   void Clear();
   void SetPulseShapeFlags(HBHERecHit& hbhe, const HBHEDataFrame &digi,
      const HcalCoder &coder, const HcalCalibrations &calib);
   void Initialize();
private:
   double mMinimumChargeThreshold;
   int mTrianglePeakTS;
   std::vector<double>  mCharge;  // stores charge for each TS in each digi
   // the pair is defined as (threshold, cut position)
   std::vector<std::pair<double, double> > mLambdaLinearCut;
   std::vector<std::pair<double, double> > mLambdaRMS8MaxCut;
   std::vector<std::pair<double, double> > mLeftSlopeCut;
   std::vector<std::pair<double, double> > mRightSlopeCut;
   std::vector<std::pair<double, double> > mRightSlopeSmallCut;
   bool mUseDualFit;
   bool mTriangleIgnoreSlow;
   std::vector<double> CumulativeIdealPulse;
private:
   TriangleFitResult PerformTriangleFit(const std::vector<double> &Charge);
   double PerformNominalFit(const std::vector<double> &Charge);
   double PerformDualNominalFit(const std::vector<double> &Charge);
   double DualNominalFitSingleTry(const std::vector<double> &Charge, int Offset, int Distance);
   double CalculateRMS8Max(const std::vector<double> &Charge);
   double PerformLinearFit(const std::vector<double> &Charge);
private:
   bool CheckPassFilter(double Charge, double Discriminant, std::vector<std::pair<double, double> > &Cuts,
      int Side);
};
//---------------------------------------------------------------------------
struct TriangleFitResult
{
   double Chi2;
   double LeftSlope;
   double RightSlope;
};
//---------------------------------------------------------------------------
#endif

