#ifndef CastorCalibrationQIEData_h
#define CastorCalibrationQIEData_h

/** 
\class CastorCalibrationQIEData
\author Panos Katsas (UoA)
POOL object to store calibration mode QIE parameters
$Id
*/

#include <vector>
#include <algorithm>

#include "CondFormats/CastorObjects/interface/CastorCalibrationQIECoder.h"
#include "DataFormats/DetId/interface/DetId.h"


// 
class CastorCalibrationQIEData {
 public:
   
  CastorCalibrationQIEData();
  ~CastorCalibrationQIEData();

   /// get QIE parameters
   const CastorCalibrationQIECoder* getCoder (DetId fId) const;
   // get list of all available channels
   std::vector<DetId> getAllChannels () const;
   // check if data are sorted
   bool sorted () const {return mSorted;}
   // fill values [capid][range]
   bool addCoder (DetId fId, const CastorCalibrationQIECoder& fCoder);
   // sort values by channelId  
   void sort ();
  typedef CastorCalibrationQIECoder Item;
  typedef std::vector <Item> Container;
 private:
   Container mItems;
   bool mSorted;
};

#endif
