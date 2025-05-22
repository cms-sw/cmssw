#ifndef DataFormats_TICLReco_MtdHostCollection_h
#define DataFormats_TICLReco_MtdHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/TICLReco/interface/MtdSoA.h"

// MtdSoA in host memory
using MtdHostCollection = PortableHostCollection<MtdSoA>;
using MtdHostCollectionView = PortableHostCollection<MtdSoA>::View;
using MtdHostCollectionConstView = PortableHostCollection<MtdSoA>::ConstView;

#endif
