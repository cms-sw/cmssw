#ifndef HFTIMINGTRUSTFLAG_GUARD_H
#define HFTIMINGTRUSTFLAG_GUARD_H

#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"

/** HFTimingTrustFlag
    Class sets the HF timing status bits according to
    uncertainty on timing estimation
       
    $Date: 2009/08/14 19:40:42 $
    \author I. Vodopiyanov -- Florida Institute of technology
*/
   
class HFTimingTrustFlag {
 public:
  HFTimingTrustFlag();
  HFTimingTrustFlag(int level1, int level2);
  ~HFTimingTrustFlag();
  
  void setHFTimingTrustFlag(HFRecHit&   rechit, const HFDataFrame&   digi);

 private:
  int HFTimingTrustLevel1_, HFTimingTrustLevel2_;

};

#endif
