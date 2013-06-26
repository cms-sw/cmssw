#ifndef HCALSIMPLERECALGO_H
#define HCALSIMPLERECALGO_H 1

#include "DataFormats/HcalDigi/interface/HcalUpgradeDataFrame.h"
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
#include "CalibCalorimetry/HcalAlgos/interface/HcalPulseContainmentManager.h"
#include <memory>

/** \class HcalSimpleRecAlgo

   This class reconstructs RecHits from Digis for HBHE, HF, and HO by addition
   of selected time samples, pedestal subtraction, and gain application. The
   time of the hit is reconstructed using a weighted peak bin calculation
   supplemented by precise time lookup table. A consumer of this class also
   has the option of correcting the reconstructed time for energy-dependent
   time slew associated with the QIE.
    
   $Date: 2013/04/26 15:49:44 $
   $Revision: 1.18 $
   \author J. Mans - Minnesota
*/
class HcalSimpleRecAlgo {
public:
  /** Full featured constructor for HB/HE and HO (HPD-based detectors) */
  HcalSimpleRecAlgo(bool correctForTimeslew, 
		    bool correctForContainment, float fixedPhaseNs);
  /** Simple constructor for PMT-based detectors */
  HcalSimpleRecAlgo();
  void beginRun(edm::EventSetup const & es);
  void endRun();

  void initPulseCorr(int toadd); 

  // set RecoParams channel-by-channel.
  void setRecoParams(bool correctForTimeslew, bool correctForPulse, bool setLeakCorrection, int pileupCleaningID, float phaseNS);

  // ugly hack related to HB- e-dependent corrections
  void setForData();
  // usage of leak correction 
  void setLeakCorrection();

  HBHERecHit reconstruct(const HBHEDataFrame& digi, int first, int toadd, const HcalCoder& coder, const HcalCalibrations& calibs) const;
  HBHERecHit reconstructHBHEUpgrade(const HcalUpgradeDataFrame& digi,  int first, int toadd, const HcalCoder& coder, const HcalCalibrations& calibs) const;

  HFRecHit reconstruct(const HFDataFrame& digi,  int first, int toadd, const HcalCoder& coder, const HcalCalibrations& calibs) const;
  HFRecHit reconstructHFUpgrade(const HcalUpgradeDataFrame& digi,  int first, int toadd, const HcalCoder& coder, const HcalCalibrations& calibs) const;

  HORecHit reconstruct(const HODataFrame& digi,  int first, int toadd, const HcalCoder& coder, const HcalCalibrations& calibs) const;
  HcalCalibRecHit reconstruct(const HcalCalibDataFrame& digi,  int first, int toadd, const HcalCoder& coder, const HcalCalibrations& calibs) const;



private:
  bool correctForTimeslew_;
  bool correctForPulse_;
  float phaseNS_;
  std::auto_ptr<HcalPulseContainmentManager> pulseCorr_;
  bool setForData_;
  bool setLeakCorrection_;
  int pileupCleaningID_;
};

#endif
