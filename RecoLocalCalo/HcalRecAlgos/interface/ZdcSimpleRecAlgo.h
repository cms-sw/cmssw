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
    
   $Date: 2011/11/10 10:15:17 $
   $Revision: 1.4 $
   \author E. Garcia CSU &  J. Gomez UMD
*/

class ZdcSimpleRecAlgo {
public:
  /** Full featured constructor for ZDC */
  ZdcSimpleRecAlgo(bool correctForTimeslew, 
		   bool correctForContainment, float fixedPhaseNs, int recoMethod, int lowGainOffset, double lowGainFrac);
  /** Simple constructor for PMT-based detectors */
  ZdcSimpleRecAlgo(int recoMethod);
    void initPulseCorr(int toadd); 
  ZDCRecHit reconstruct(const ZDCDataFrame& digi, const std::vector<unsigned int>& myNoiseTS, const std::vector<unsigned int>& mySignalTS, const HcalCoder& coder, const HcalCalibrations& calibs) const;
  HcalCalibRecHit reconstruct(const HcalCalibDataFrame& digi, const std::vector<unsigned int>& myNoiseTS, const std::vector<unsigned int>& mySignalTS, const HcalCoder& coder, const HcalCalibrations& calibs) const;
private:
  int recoMethod_;
  bool correctForTimeslew_;
   bool correctForPulse_;
     float phaseNS_;
   // new lowGainEnergy variables
   int lowGainOffset_;
   double lowGainFrac_;
  std::auto_ptr<HcalPulseContainmentCorrection> pulseCorr_;
};

#endif
