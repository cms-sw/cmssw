#ifndef CondFormats_SiPixelObjects_interface_alpaka_SiPixelGainCalibrationForHLTDevice_h
#define CondFormats_SiPixelObjects_interface_alpaka_SiPixelGainCalibrationForHLTDevice_h

#include "CondFormats/SiPixelObjects/interface/SiPixelGainCalibrationForHLTLayout.h"
#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using SiPixelGainCalibrationForHLTDevice = PortableCollection<SiPixelGainCalibrationForHLTSoA>;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#endif  // CondFormats_SiPixelObjects_interface_alpaka_SiPixelGainCalibrationForHLTDevice_h
