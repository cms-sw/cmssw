#ifndef CASTORSIMPLERECALGO_H
#define CASTORSIMPLERECALGO_H 1

#include "DataFormats/HcalDigi/interface/CastorDataFrame.h"
#include "DataFormats/HcalRecHit/interface/CastorRecHit.h"
#include "CalibFormats/CastorObjects/interface/CastorCoder.h"
#include "CalibFormats/CastorObjects/interface/CastorCalibrations.h"
#include "CalibCalorimetry/CastorCalib/interface/CastorPulseContainmentCorrection.h"
#include <memory>

/** \class CastorSimpleRecAlgo

   This class reconstructs RecHits from Digis for CASTOR by addition
   of selected time samples, pedestal subtraction, and gain application. The
   time of the hit is reconstructed using a weighted peak bin calculation
   supplemented by precise time lookup table. A consumer of this class also
   has the option of correcting the reconstructed time for energy-dependent
   time slew associated with the QIE.

   \author P. Katsas (Univ. of Athens) 
*/
class CastorSimpleRecAlgo {
public:
  /** Full featured constructor for HB/HE and HO (HPD-based detectors) */
  CastorSimpleRecAlgo(int firstSample, int samplesToAdd, bool correctForTimeslew, 
		    bool correctForContainment, float fixedPhaseNs);
  /** Simple constructor for PMT-based detectors */
  CastorSimpleRecAlgo(int firstSample, int samplesToAdd);

  CastorRecHit reconstruct(const CastorDataFrame& digi, const CastorCoder& coder, const CastorCalibrations& calibs) const;

  // sets rechit saturation status bit on if ADC count is >= maxADCvalue
  void checkADCSaturation(CastorRecHit& rechit, const CastorDataFrame& digi, const int& maxADCvalue) const;

  //++++ Saturation Correction +++++
  // recover pulse shape if ADC count is >= masADCvalue
  void recoverADCSaturation(CastorRecHit& rechit, const CastorCoder& coder, const CastorCalibrations& calibs,
			    const CastorDataFrame& digi, const int& maxADCvalue, const double& satCorrConst) const;
  
  

  void resetTimeSamples(int f,int t){
    firstSample_=f;
    samplesToAdd_=t;
  }
private:
  int firstSample_, samplesToAdd_;
  bool correctForTimeslew_;
  std::auto_ptr<CastorPulseContainmentCorrection> pulseCorr_;
};

#endif
