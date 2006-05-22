#ifndef HcalCalibrationQIEData_h
#define HcalCalibrationQIEData_h

/** 
\class HcalCalibrationQIEData
\author Fedor Ratnikov (UMd)
POOL object to store calibration mode QIE parameters
$Id
*/

#include <vector>
#include <algorithm>

#include "CondFormats/HcalObjects/interface/HcalCalibrationQIECoder.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


// 
class HcalCalibrationQIEData {
 public:
   
  HcalCalibrationQIEData();
  ~HcalCalibrationQIEData();

   /// get QIE parameters
   const HcalCalibrationQIECoder* getCoder (HcalDetId fId) const;
   // get list of all available channels
   std::vector<HcalDetId> getAllChannels () const;
   // check if data are sorted
   bool sorted () const {return mSorted;}
   // fill values [capid][range]
   bool addCoder (HcalDetId fId, const HcalCalibrationQIECoder& fCoder);
   // sort values by channelId  
   void sort ();
  typedef HcalCalibrationQIECoder Item;
  typedef std::vector <Item> Container;
 private:
   Container mItems;
   bool mSorted;
};

#endif
