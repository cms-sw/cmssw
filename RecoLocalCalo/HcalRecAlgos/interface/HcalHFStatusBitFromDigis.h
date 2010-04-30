#ifndef HCALHFSTATUSFROMDIGIS_H
#define HCALHFSTATUSFROMDIGIS_H 1

#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"
#include "CalibFormats/HcalObjects/interface/HcalCalibrations.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

/** \class HcalHFStatusBitFromDigis
    
   This class sets status bit in the status words for the revised CaloRecHit objets according to informatino from the digi associated to the hit.
    
   $Date: 2010/04/28 17:48:58 $
   $Revision: 1.5 $
   \author J. Temple -- University of Maryland and E. Yazgan
*/

class HcalHFStatusBitFromDigis {
public:
  /** Full featured constructor for HB/HE and HO (HPD-based detectors) */
  HcalHFStatusBitFromDigis();
  HcalHFStatusBitFromDigis(int recoFirstSample,
			   int recoSamplesToAdd,
			   const edm::ParameterSet& HFDigiTimeParams,
			   const edm::ParameterSet& HFTimeInWindowParams);
  // Destructor
  ~HcalHFStatusBitFromDigis();

  // The important stuff!  Methods for setting the status flag values
  void hfSetFlagFromDigi(HFRecHit& hf, const HFDataFrame& digi, const HcalCalibrations& calib);
  

  double bit(){return HcalCaloFlagLabels::HFDigiTime;}
  double threshold(){return minthreshold_;}

  // setter functions
  void setthreshold(double x){minthreshold_=x; return;}

private:
  // variables for cfg files
  double minthreshold_;
  // Reco Window
  int recoFirstSample_;
  int recoSamplesToAdd_;
  // Special window for Igor's algorithm (not necessarily the same as reco window)
  int firstSample_;
  int samplesToAdd_;
  int expectedPeak_;
  
  // Coefficients used to determine energy ratio threshold:
  // E_peak/(Etotal) > coef0_-exp(coef1_+coef2_*Energy)
  double coef0_;
  double coef1_;
  double coef2_;

  // Store minimum/maximum allowed rechit times.  
  // (Times outside window are flagged)
  double HFwindowEthresh_;
  std::vector<double> HFwindowMinTime_;
  std::vector<double> HFwindowMaxTime_;
};

#endif
