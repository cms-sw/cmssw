#ifndef CastorQIEData_h
#define CastorQIEData_h

/** 
\class CastorQIEData
\author Fedor Ratnikov (UMd), with changes by Radek Ofierzynski 
   (preserve backwards compatibility of methods for this release)
Modified by L.Mundim (Mar/2009)
POOL object to store QIE parameters
$Author: ratnikov
$Date: 2009/03/26 18:03:15 $
$Revision: 1.2 $
*/

#include <vector>
#include <algorithm>

#include "CondFormats/CastorObjects/interface/CastorCondObjectContainer.h"
#include "CondFormats/CastorObjects/interface/CastorQIEShape.h"
#include "CondFormats/CastorObjects/interface/CastorQIECoder.h"
#include "DataFormats/DetId/interface/DetId.h"

namespace
{
  CastorQIEShape shape_;
}

class CastorQIEData: public CastorCondObjectContainer<CastorQIECoder>
{
 public:

  // constructor, destructor, and all methods stay the same
 CastorQIEData():CastorCondObjectContainer<CastorQIECoder>() {}

  /// get basic shape
  //   const CastorQIEShape& getShape () const {return mShape;}
  const CastorQIEShape& getShape () const { return shape_;}
  /// get QIE parameters
  const CastorQIECoder* getCoder (DetId fId) const { return getValues(fId); }
  // check if data are sorted - remove in the next version
  bool sorted () const { return true; }
  // fill values [capid][range]
  //bool addCoder (const CastorQIECoder& fCoder, bool h2mode_ = false) { return addValues(fCoder, h2mode_); }
  bool addCoder (const CastorQIECoder& fCoder) { return addValues(fCoder); }
  // sort values by channelId - remove in the next version  
  void sort () {}
  
  std::string myname() const {return (std::string)"CastorQIEData";}

  //not needed/not used  CastorQIEData(const CastorQIEData&);

};

#endif
