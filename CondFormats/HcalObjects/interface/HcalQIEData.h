#ifndef HcalQIEData_h
#define HcalQIEData_h

/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd)
POOL object to store QIE parameters
$Author: ratnikov
$Date: 2005/10/28 01:37:10 $
$Revision: 1.2 $
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
   ~HcalQIEData();

   /// get basic shape
   const HcalQIEShape& getShape () const {return mShape;}
   /// get QIE parameters
   const HcalQIECoder* getCoder (HcalDetId fId) const;
   // get list of all available channels
   std::vector<HcalDetId> getAllChannels () const;
   // check if data are sorted
   bool sorted () const {return mSorted;}
   // fill shape
   bool setShape (const float fLowEdges [32]);
   // fill values [capid][range]
   bool addCoder (HcalDetId fId, const HcalQIECoder& fCoder);
   // sort values by channelId  
   void sort ();
   HcalQIEShape mShape;
   std::vector<HcalQIECoder> mItems;
   bool mSorted;
};

#endif
