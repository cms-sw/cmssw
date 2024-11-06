#ifndef ZDCSIMPLERECALGO_RUN3_H
#define ZDCSIMPLERECALGO_RUN3_H 1

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
#include "CalibFormats/HcalObjects/interface/HcalCalibrationWidths.h"
#include "CondFormats/HcalObjects/interface/HcalPedestal.h"
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentCorrection.h"
#include <memory>

#include "DataFormats/HcalDigi/interface/QIE10DataFrame.h"

/** \class ZdcSimpleRecAlgo_Run3

   This class reconstructs RecHits from Digis for ZDC  by addition
   of selected time samples, pedestal subtraction, and gain application. The
   time of the hit is reconstructed using a weighted peak bin calculation
   supplemented by precise time lookup table. A consumer of this class also
   has the option of correcting the reconstructed time for energy-dependent
   time slew associated with the QIE.

   A sencon method based on a based on a event by event substraction is also
   implelented. signal = (S4 + S5 - 2*(S1+S2+S3 + S7+S8+S9+S10))*(ft-Gev constant)
   where SN is the signal in the nth time slice
    
   \author E. Garcia CSU &  J. Gomez UMD
*/

class ZdcSimpleRecAlgo_Run3 {
public:
  /** Simple constructor for PMT-based detectors */
  ZdcSimpleRecAlgo_Run3(int recoMethod);
  void initCorrectionMethod(const int method, const int ZdcSection);
  void initTemplateFit(const std::vector<unsigned int>& bxTs,
                       const std::vector<double>& chargeRatios,
                       const int nTs,
                       const int ZdcSection);
  void initRatioSubtraction(const float ratio, const float frac, const int ZdcSection);

  ZDCRecHit reco0(const QIE10DataFrame& digi,
                  const HcalCoder& coder,
                  const HcalCalibrations& calibs,
                  const HcalPedestal& effPeds,
                  const std::vector<unsigned int>& myNoiseTS,
                  const std::vector<unsigned int>& mySignalTS) const;
  // reco method currently used to match L1 Trigger LUT energy formula
  ZDCRecHit reconstruct(const QIE10DataFrame& digi,
                        const std::vector<unsigned int>& myNoiseTS,
                        const std::vector<unsigned int>& mySignalTS,
                        const HcalCoder& coder,
                        const HcalCalibrations& calibs,
                        const HcalPedestal& effPeds) const;

private:
  int recoMethod_;
  int nTs_;
  std::map<int, std::vector<double>> templateFitValues_;  // Values[ZdcSection][Ts]
  std::map<int, bool> templateFitValid_;                  // Values[ZdcSection]
  std::map<int, float> ootpuRatio_;                       // Values[ZdcSection]
  std::map<int, float> ootpuFrac_;                        // Values[ZdcSection]
  std::map<int, int> correctionMethod_;                   // Values[ZdcSection]
};

#endif
