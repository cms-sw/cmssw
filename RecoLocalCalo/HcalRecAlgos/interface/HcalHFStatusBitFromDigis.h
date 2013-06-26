#ifndef HCALHFSTATUSFROMDIGIS_H
#define HCALHFSTATUSFROMDIGIS_H 1

#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CalibFormats/HcalObjects/interface/HcalCoderDb.h"

/** \class HcalHFStatusBitFromDigis
    
   This class sets status bit in the status words for the revised CaloRecHit objets according to informatino from the digi associated to the hit.
    
   $Date: 2013/05/30 21:37:33 $
   $Revision: 1.16 $
   \author J. Temple -- University of Maryland and E. Yazgan
*/

class HcalHFStatusBitFromDigis {
public:
  /** Full featured constructor for HB/HE and HO (HPD-based detectors) */
  HcalHFStatusBitFromDigis();
  HcalHFStatusBitFromDigis(const edm::ParameterSet& HFDigiTimeParams,
			   const edm::ParameterSet& HFTimeInWindowParams);
  // Destructor
  ~HcalHFStatusBitFromDigis();

  // The important stuff!  Methods for setting the status flag values
  void hfSetFlagFromDigi(HFRecHit& hf, const HFDataFrame& digi,
			 const HcalCoder& coder,
			 const HcalCalibrations& calib);
  void resetParamsFromDB(int firstSample, int samplesToAdd, int expectedPeak, double minthreshold, const std::vector<double>& coef);
  void resetFlagTimeSamples(int firstSample, int samplesToAdd, int expectedPeak);

private:

  // variables for cfg files

  // VARIABLES FOR SETTING HFDigiTime FLAG
  double minthreshold_;
  // Reco Window
  int recoFirstSample_;
  int recoSamplesToAdd_;
  // Special window for Igor's algorithm (not necessarily the same as reco window)
  int firstSample_;
  int samplesToAdd_;
  int expectedPeak_;
  
  // Coefficients used to determine energy ratio threshold:
  // E_peak/(Etotal) > coef0_-exp(coef1_+coef2_*Energy+coef3_*E^2+...)
  std::vector<double> coef_;


  // VARIABLES FOR SETTING HFInTimeWindow FLAG
  double HFlongwindowEthresh_;
  std::vector<double> HFlongwindowMinTime_;
  std::vector<double> HFlongwindowMaxTime_;
  double HFshortwindowEthresh_;
  std::vector<double> HFshortwindowMinTime_;
  std::vector<double> HFshortwindowMaxTime_;
};

#endif
