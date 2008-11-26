#ifndef HCALHFSTATUSBITFROMRECHITS_H
#define HCALHFSTATUSBITFROMRECHITS_H 1

#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalRecHit/interface/HcalRecHitCollections.h"

/** \class HcalHFStatusBitFromRecHits
    
   This class sets status bit in the status words for the revised CaloRecHit objets using comparisons between the rec hit energies of long and short fibers for a given HF (ieat, iphi)
    
   $Date: 2008/03/11 17:02:07 $
   $Revision: 1.1 $
   \author J. Temple -- University of Maryland and E. Yazgan
*/

class HcalHFStatusBitFromRecHits {
public:
  /** Full featured constructor for HB/HE and HO (HPD-based detectors) */
  HcalHFStatusBitFromRecHits();
  HcalHFStatusBitFromRecHits(double HFlongshort, int bit=1); 
  
  // Destructor
  ~HcalHFStatusBitFromRecHits();

  // The important stuff!  Methods for setting the status flag values
  void hfSetFlagFromRecHits(HFRecHitCollection& rec);

  // getter functions
  double hflongshortratio(){return HFlongshortratio_;}
  double bit(){return bit_;}
  // setter functions
  void sethflongshortratio(double x){HFlongshortratio_=x; return;}

private:
  // variables for cfg files
  double HFlongshortratio_;
  int bit_; // defines which bit to set in status word
};

#endif
