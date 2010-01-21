#ifndef HCALHFSTATUSFROMDIGIS_H
#define HCALHFSTATUSFROMDIGIS_H 1

#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalCaloFlagLabels.h"

/** \class HcalHFStatusBitFromDigis
    
   This class sets status bit in the status words for the revised CaloRecHit objets according to informatino from the digi associated to the hit.
    
   $Date: 2009/03/27 14:46:59 $
   $Revision: 1.2 $
   \author J. Temple -- University of Maryland and E. Yazgan
*/

class HcalHFStatusBitFromDigis {
public:
  /** Full featured constructor for HB/HE and HO (HPD-based detectors) */
  HcalHFStatusBitFromDigis();
  HcalHFStatusBitFromDigis(int HFpulsetimemin,int HFpulsetimemax, double HFratiobefore, double HFratioafter, int adcthreshold=10, int firstSample=0, int samplesToAdd=10); 
  
  // Destructor
  ~HcalHFStatusBitFromDigis();

  // The important stuff!  Methods for setting the status flag values
  void hfSetFlagFromDigi(HFRecHit& hf, const HFDataFrame& digi);
  
  // getter functions -- would we ever want to do this?
  int hfpulsetimemin(){return HFpulsetimemin_;}
  int hfpulsetimemax(){return HFpulsetimemax_;}
  double hfratio_beforepeak(){return HFratio_beforepeak_;}
  double hfratio_afterpeak(){return HFratio_afterpeak_;}
  double bit(){return HcalCaloFlagLabels::HFDigiTime;}
  double threshold(){return adcthreshold_;}

  // setter functions
  void sethfpulsetimemin(int x){HFpulsetimemin_=x; return;}
  void sethfpulsetimemax(int x){HFpulsetimemax_=x; return;}
  void sethfratio_beforepeak(double x){HFratio_beforepeak_=x; return;}
  void sethfratio_afterpeak(double x){HFratio_afterpeak_=x; return;}
  void setthreshold(double x){adcthreshold_=x; return;}

private:
  // variables for cfg files
  int HFpulsetimemin_, HFpulsetimemax_;
  double HFratio_beforepeak_, HFratio_afterpeak_;
  double adcthreshold_;
  int firstSample_;
  int samplesToAdd_;
};

#endif
