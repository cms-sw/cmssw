#ifndef DataFormats_HGCalReco_MtdHostCollection_h
#define DataFormats_HGCalReco_MtdHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/HGCalReco/interface/MtdSoA.h"

// MtdSoA in host memory
using MtdHostCollection = PortableHostCollection<MtdSoA>;
using MtdHostCollectionView = PortableHostCollection<MtdSoA>::View;
using MtdHostCollectionConstView = PortableHostCollection<MtdSoA>::ConstView;

#endif
