#ifndef DataFormats_FTLDigiSoA_interface_BTLDigiCollection_h
#define DataFormats_FTLDigiSoA_interface_BTLDigiCollection_h

#include "DataFormats/FTLDigiSoA/interface/BTLDigiSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace btldigi {

  using BTLDigiHostCollection = PortableHostCollection<BTLDigiSoA>;

}  //namespace btldigi

#endif  // DataFormats_FTLDigi_interface_BTLDigiCollection_h
