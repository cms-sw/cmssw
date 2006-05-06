#ifndef HcalQIEData_h
#define HcalQIEData_h

/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store QIE parameters
$Author: ratnikov
$Date: 2005/12/15 23:38:04 $
$Revision: 1.3 $
*/

#include <vector>
#include <algorithm>

#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "DataFormats/HcalDetId/interface/HcalDetId.h"


// 
class HcalQIEData {
 public:
   
  HcalQIEData();
  HcalQIEData(const HcalQIEData&);
   ~HcalQIEData();

   /// get basic shape
   //   const HcalQIEShape& getShape () const {return mShape;}
   const HcalQIEShape& getShape () const;
   /// get QIE parameters
   const HcalQIECoder* getCoder (HcalDetId fId) const;
   // get list of all available channels
   std::vector<HcalDetId> getAllChannels () const;
   // check if data are sorted
   bool sorted () const {return mSorted;}
   // fill values [capid][range]
   bool addCoder (HcalDetId fId, const HcalQIECoder& fCoder);
   // sort values by channelId  
   void sort ();
  typedef HcalQIECoder Item;
  typedef std::vector <Item> Container;
 private:
   Container mItems;
   bool mSorted;
};

#endif
