#ifndef HCALADCSATURATIONFLAG_GUARD_H
#define HCALADCSATURATIONFLAG_GUARD_H


#include "DataFormats/HcalDigi/interface/HBHEDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HBHERecHit.h"
#include "DataFormats/HcalDigi/interface/HODataFrame.h"
#include "DataFormats/HcalRecHit/interface/HORecHit.h"
#include "DataFormats/HcalDigi/interface/HFDataFrame.h"
#include "DataFormats/HcalRecHit/interface/HFRecHit.h"
#include "DataFormats/HcalDigi/interface/ZDCDataFrame.h"
#include "DataFormats/HcalRecHit/interface/ZDCRecHit.h"

/** HcalADCSaturationFlag
    Class sets the Saturation status bit if the ADC count for any time slice 
    within a digi is >= a certain value (SaturationLevel_).  This value is 
    user-configurable, but perhaps it should be hard-coded, as we know the
    QIE's have 7 bits (for a maximum ADC value of 2^7=127).  
    Is it better to hard-code to 127, or to allow the user to change it?
       
    $Date: 2010/10/22 03:02:53 $
    $Revision: 1.2 $
    \author J. Temple -- University of Maryland
*/
   

class HcalADCSaturationFlag {
 public:
  HcalADCSaturationFlag();
  HcalADCSaturationFlag(int level);
  ~HcalADCSaturationFlag();
  
  void setSaturationFlag(HBHERecHit& rechit, const HBHEDataFrame& digi);
  void setSaturationFlag(HORecHit&   rechit, const HODataFrame&   digi);
  void setSaturationFlag(HFRecHit&   rechit, const HFDataFrame&   digi);
  void setSaturationFlag(ZDCRecHit&   rechit, const ZDCDataFrame&   digi);

 private:
  int SaturationLevel_;

};

#endif
