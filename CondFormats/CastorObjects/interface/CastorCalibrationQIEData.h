#ifndef CastorCalibrationQIEData_h
#define CastorCalibrationQIEData_h

/** 
\class CastorCalibrationQIEData
\author Fedor Ratnikov (UMd), with changes by Radek Ofierzynski 
   (preserve backwards compatibility of methods for this release)
   Adapted for CASTOR by L. Mundim

POOL object to store calibration mode QIE parameters
$Id
*/

#include <vector>
#include <algorithm>

#include "CondFormats/CastorObjects/interface/CastorCondObjectContainer.h"

#include "CondFormats/CastorObjects/interface/CastorCalibrationQIECoder.h"
#include "DataFormats/DetId/interface/DetId.h"


class CastorCalibrationQIEData: public CastorCondObjectContainer<CastorCalibrationQIECoder>
{
 public:
   
  /// get QIE parameters
  const CastorCalibrationQIECoder* getCoder (DetId fId) const { return getValues(fId); }
  // check if data are sorted
  bool sorted () const {return true;}
  // fill values [capid][range]
  bool addCoder (const CastorCalibrationQIECoder& fCoder) { return addValues(fCoder); }
   // sort values by channelId  
  void sort () {}

};

#endif
