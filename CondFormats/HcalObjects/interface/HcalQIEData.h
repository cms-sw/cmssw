#ifndef HcalQIEData_h
#define HcalQIEData_h

/** 
\class HcalQIEData
\author Fedor Ratnikov (UMd), with changes by Radek Ofierzynski 
   (preserve backwards compatibility of methods for this release)
POOL object to store QIE parameters
$Author: ratnikov
$Date: 2009/03/24 16:05:32 $
$Revision: 1.10 $
*/

#include <vector>
#include <algorithm>

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoder.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace
{
  HcalQIEShape shape_;
}

class HcalQIEData: public HcalCondObjectContainer<HcalQIECoder>
{
 public:

  // constructor, destructor, and all methods stay the same
 HcalQIEData():HcalCondObjectContainer<HcalQIECoder>() {}

  /// get basic shape
  //   const HcalQIEShape& getShape () const {return mShape;}
  const HcalQIEShape& getShape () const { return shape_;}
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

};

#endif
