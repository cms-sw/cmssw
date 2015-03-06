#ifndef HCALHFSTATUSBITFROMRECHITS_H
#define HCALHFSTATUSBITFROMRECHITS_H 1

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"
#include "DataFormats/METReco/interface/HcalCaloFlagLabels.h"
#include "CondFormats/HcalObjects/interface/HcalChannelQuality.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputer.h"
#include "RecoLocalCalo/HcalRecAlgos/interface/HcalSeverityLevelComputerRcd.h"


/** \class HcalHFStatusBitFromRecHits
    
   This class sets status bit in the status words for the revised CaloRecHit objets using comparisons between the rec hit energies of long and short fibers for a given HF (ieat, iphi)
    
   \author J. Temple -- University of Maryland and E. Yazgan
*/

class HcalHFStatusBitFromRecHits {
public:
  /** Full featured constructor for HB/HE and HO (HPD-based detectors) */
  HcalHFStatusBitFromRecHits();
  HcalHFStatusBitFromRecHits(double shortR, double shortET, double shortE,
			     double longR, double longET, double longE);
  
  // Destructor
  ~HcalHFStatusBitFromRecHits();

  // The important stuff!  Methods for setting the status flag values
  void hfSetFlagFromRecHits(HFRecHitCollection& rec,
			    HcalChannelQuality* myqual, 
			    const HcalSeverityLevelComputer* mySeverity);

  // getter functions
  double long_hflongshortratio(){return long_HFlongshortratio_;}
  double long_energythreshold(){return long_thresholdEnergy_;}
  double long_ETthreshold(){return long_thresholdET_;}
  double short_hflongshortratio(){return short_HFlongshortratio_;}
  double short_energythreshold(){return short_thresholdEnergy_;}
  double short_ETthreshold(){return short_thresholdET_;}

  double bit(){return HcalCaloFlagLabels::HFLongShort;}

  // setter functions
  void set_long_hflongshortratio(double x){long_HFlongshortratio_=x; return;}
  void set_long_energythreshold(double x){long_thresholdEnergy_=x; return;}
  void set_long_ETthreshold(double x){long_thresholdET_=x; return;}
  void set_short_hflongshortratio(double x){short_HFlongshortratio_=x; return;}
  void set_short_energythreshold(double x){short_thresholdEnergy_=x; return;}
  void set_short_ETthreshold(double x){short_thresholdET_=x; return;}

private:
  // variables for cfg files
  double long_HFlongshortratio_;
  double long_thresholdET_;  // minimum energy needed before the noise algorithm is run
  double long_thresholdEnergy_;
  double short_HFlongshortratio_;
  double short_thresholdET_;  // minimum energy needed before the noise algorithm is run
  double short_thresholdEnergy_;
};

#endif
