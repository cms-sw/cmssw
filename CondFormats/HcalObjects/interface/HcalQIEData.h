#ifndef HcalQIEData_h
#define HcalQIEData_h

/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd), with changes by Radek Ofierzynski 
   (preserve backwards compatibility of methods for this release)
POOL object to store QIE parameters
$Author: ratnikov
$Date: 2012/03/29 16:20:12 $
$Revision: 1.12 $
*/

#include <vector>
#include <algorithm>

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "DataFormats/DetId/interface/DetId.h"


class HcalQIEData: public HcalCondObjectContainer<HcalQIECoder>
{
 public:

  // constructor, destructor, and all methods stay the same
  HcalQIEData();

  /// get basic shape
  //   const HcalQIEShape& getShape () const {return mShape;}
   const HcalQIEShape& getShape (DetId fId) const { return mShape[getCoder(fId)->qieIndex()];}
   const HcalQIEShape& getShape (const HcalQIECoder* coder) const { return mShape[coder->qieIndex()];}
  /// get QIE parameters
  const HcalQIECoder* getCoder (DetId fId) const { return getValues(fId); }
  // check if data are sorted - remove in the next version
  bool sorted () const { return true; }
  // fill values [capid][range]
  bool addCoder (const HcalQIECoder& fCoder, bool h2mode_ = false) { return addValues(fCoder, h2mode_); }
  // sort values by channelId - remove in the next version  
  void sort () {}
  
  std::string myname() const {return (std::string)"HcalQIEData";}

  //not needed/not used  HcalQIEData(const HcalQIEData&);

 private:
  HcalQIEShape mShape[2];

};

#endif
