#ifndef ZDCSIMPLERECALGO_H
#define ZDCSIMPLERECALGO_H 1

#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalDigi/interface/HcalCalibDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalCalibRecHit.h"
#include "CalibFormats/HcalObjects/interface/HcalCoder.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentCorrection.h"
#include <memory>

/** \class ZdcSimpleRecAlgo

   This class reconstructs RecHits from Digis for ZDC  by addition
   of selected time samples, pedestal subtraction, and gain application. The
   time of the hit is reconstructed using a weighted peak bin calculation
   supplemented by precise time lookup table. A consumer of this class also
   has the option of correcting the reconstructed time for energy-dependent
   time slew associated with the QIE.

   A sencon method based on a based on a event by event substraction is also
   implelented. signal = (S4 + S5 - 2*(S1+S2+S3 + S7+S8+S9+S10))*(ft-Gev constant)
   where SN is the signal in the nth time slice
    
   $Date: 2010/01/21 14:28:18 $
   $Revision: 1.1 $
   \author E. Garcia CSU &  J. Gomez UMD
*/

class ZdcSimpleRecAlgo {
public:
  /** Full featured constructor for ZDC */
  ZdcSimpleRecAlgo(int firstSample, int firstNoise, int samplesToAdd, bool correctForTimeslew, 
		   bool correctForContainment, float fixedPhaseNs, int recoMethod);
  /** Simple constructor for PMT-based detectors */
  ZdcSimpleRecAlgo(int firstSample, int firstNoise, int samplesToAdd, int recoMethod);
  ZDCRecHit reconstruct(const ZDCDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const;
  HcalCalibRecHit reconstruct(const HcalCalibDataFrame& digi, const HcalCoder& coder, const HcalCalibrations& calibs) const;
private:
  int firstSample_, firstNoise_, samplesToAdd_, recoMethod_;
  bool correctForTimeslew_;
  std::auto_ptr<HcalPulseContainmentCorrection> pulseCorr_;
};

#endif
