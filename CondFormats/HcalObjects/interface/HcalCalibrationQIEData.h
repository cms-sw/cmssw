#ifndef HcalCalibrationQIEData_h
#define HcalCalibrationQIEData_h

/** 
\class HcalCalibrationQIEData
\author Fedor Ratnikov (UMd), with changes by Radek Ofierzynski 
   (preserve backwards compatibility of methods for this release)
POOL object to store calibration mode QIE parameters
$Id
*/

#include <vector>
#include <algorithm>

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"

#include "CondFormats/HcalObjects/interface/HcalCalibrationQIECoder.h"
#include "DataFormats/DetId/interface/DetId.h"


class HcalCalibrationQIEData: public HcalCondObjectContainer<HcalCalibrationQIECoder>
{
 public:
  HcalCalibrationQIEData(const HcalTopology* ht) : HcalCondObjectContainer<HcalCalibrationQIECoder>(ht) { }
  /// get QIE parameters
  const HcalCalibrationQIECoder* getCoder (DetId fId) const { return getValues(fId); }
  // check if data are sorted
  bool sorted () const {return true;}
  // fill values [capid][range]
  bool addCoder (const HcalCalibrationQIECoder& fCoder) { return addValues(fCoder); }
   // sort values by channelId  
  void sort () {}

};

#endif
