#ifndef CastorQIEData_h
#define CastorQIEData_h

/** 
\class CastorQIEData
\author Panos Katsas (UoA)
POOL object to store QIE parameters
*/

#include <vector>
#include <algorithm>

#include "CondFormats/CastorObjects/interface/CastorQIEShape.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"
#include "DataFormats/DetId/interface/DetId.h"


// 
class CastorQIEData {
 public:
   
  CastorQIEData();
  CastorQIEData(const CastorQIEData&);
   ~CastorQIEData();

   /// get basic shape
   //   const HcalQIEShape& getShape () const {return mShape;}
   const CastorQIEShape& getShape () const;
   /// get QIE parameters
   const CastorQIECoder* getCoder (DetId fId) const;
   // get list of all available channels
   std::vector<DetId> getAllChannels () const;
   // check if data are sorted
   bool sorted () const {return mSorted;}
   // fill values [capid][range]
   bool addCoder (DetId fId, const CastorQIECoder& fCoder);
   // sort values by channelId  
   void sort ();
  typedef CastorQIECoder Item;
  typedef std::vector <Item> Container;
 private:
   Container mItems;
   bool mSorted;
};

#endif
