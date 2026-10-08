#ifndef DataFormats_FTLDigiSoA_interface_ETLDigiCollection_h
#define DataFormats_FTLDigiSoA_interface_ETLDigiCollection_h

#include "DataFormats/FTLDigiSoA/interface/ETLDigiSoA.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"

namespace etldigi {

  using ETLDigiHostCollection = PortableHostCollection<ETLDigiSoA>;
  using ETLDigiHostCollectionView = PortableHostCollection<ETLDigiSoA>::View;
  using ETLDigiHostCollectionConstView = PortableHostCollection<ETLDigiSoA>::ConstView;

}  //namespace etldigi

#endif  // DataFormats_FTLDigi_interface_ETLDigiCollection_h
