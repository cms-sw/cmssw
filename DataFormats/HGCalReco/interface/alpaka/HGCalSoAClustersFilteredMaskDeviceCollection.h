#ifndef DataFormats_HGCalReco_interface_alpaka_HGCalSoAClustersFilteredMaskDeviceCollection_h
#define DataFormats_HGCalReco_interface_alpaka_HGCalSoAClustersFilteredMaskDeviceCollection_h

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "DataFormats/HGCalReco/interface/HGCalSoAClustersFilteredMask.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {
    using HGCalSoAClustersFilteredMaskDeviceCollection = PortableCollection<HGCalSoAClustersFilteredMask>;
    using HGCalSoAClustersFilteredMaskDeviceCollectionView = HGCalSoAClustersFilteredMaskDeviceCollection::View;
    using HGCalSoAClustersFilteredMaskDeviceCollectionConstView = HGCalSoAClustersFilteredMaskDeviceCollection::ConstView;
}

#endif