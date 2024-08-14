#ifndef DataFormats_HGCalReco_interface_HGCalSoAClustersFilteredMask_h
#define DataFormats_HGCalReco_interface_HGCalSoAClustersFilteredMask_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"


GENERATE_SOA_LAYOUT(HGCalSoAClustersFilteredMaskLayout,
    SOA_COLUMN(float, mask)
)

using HGCalSoAClustersFilteredMask = HGCalSoAClustersFilteredMaskLayout<>;
using HGCalSoAClustersFilteredMaskView = HGCalSoAClustersFilteredMask::View;
using HGCalSoAClustersFilteredMaskConstView = HGCalSoAClustersFilteredMask::ConstView;

#endif