#ifndef HcalQIEDataExtended_h
#define HcalQIEDataExtended_h

/** 
\class HcalQIEDataExtended
\author Clemencia Mora based on  HcalQIEData by Fedor Ratnikov (UMd), with changes by Radek Ofierzynski 
   (preserve backwards compatibility of methods for this release)
POOL object to store QIE parameters with QIE barcodes and channel numbers
$Author: cmora
$Date: 2015/10/12 $
$Revision: 1.0 $
*/

#include "CondFormats/Serialization/interface/Serializable.h"

#include <vector>
#include <algorithm>

#include "CondFormats/HcalObjects/interface/HcalCondObjectContainer.h"
#include "CondFormats/HcalObjects/interface/HcalQIEShape.h"
#include "CondFormats/HcalObjects/interface/HcalQIECoderExtended.h"
#include "DataFormats/DetId/interface/DetId.h"


class HcalQIEDataExtended: public HcalCondObjectContainer<HcalQIECoderExtended>
{
 public:
#ifndef HCAL_COND_SUPPRESS_DEFAULT
  HcalQIEDataExtended():HcalCondObjectContainer<HcalQIECoderExtended>(0) {setupShape();}
#endif
  // constructor, destructor, and all methods stay the same
  HcalQIEDataExtended(const HcalTopology* topo):HcalCondObjectContainer<HcalQIECoderExtended>(topo) {setupShape();}

  void setupShape();  
  /// get basic shape
  //   const HcalQIEShape& getShape () const {return mShape;}
   const HcalQIEShape& getShape (DetId fId) const { return mShape[getCoder(fId)->qieIndex()];}
   const HcalQIEShape& getShape (const HcalQIECoderExtended* coder) const { return mShape[coder->qieIndex()];}
  /// get QIE parameters
  const HcalQIECoderExtended* getCoder (DetId fId) const { return getValues(fId); }
  // check if data are sorted - remove in the next version
  bool sorted () const { return true; }
  // fill values [capid][range]
  bool addCoder (const HcalQIECoderExtended& fCoder) { return addValues(fCoder); }
  // sort values by channelId - remove in the next version  
  void sort () {}
  
  std::string myname() const {return (std::string)"HcalQIEDataExtended";}

  //not needed/not used  HcalQIEDataExtended(const HcalQIEDataExtended&);

  const int getBarcode(DetId fId) const {return getCoder(fId)->getQIEbarcode();}
  const int getChannel(DetId fId) const {return getCoder(fId)->getQIEchannel();}
  //void setQIEId(DetId fId, int barcode,int channel) const {getCoder(fId)->setQIEId(barcode,channel);}


 private:
  HcalQIEShape mShape[2] COND_TRANSIENT;
 COND_SERIALIZABLE;
};

#endif
