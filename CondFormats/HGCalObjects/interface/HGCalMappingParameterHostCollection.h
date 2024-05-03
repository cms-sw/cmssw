#ifndef CondFormats_HGCalObjects_interface_HGCalMappingParameterHostCollection_h
#define CondFormats_HGCalObjects_interface_HGCalMappingParameterHostCollection_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterSoA.h"

namespace hgcal {

  // SoA with channel-level module mapping parameters in host memory:
  using HGCalMappingModuleParamHostCollection = PortableHostCollection<HGCalMappingModuleParamSoA>;

  // SoA with channel-level cell mapping parameters in host memory for both Si and SiPM channels:
  using HGCalMappingCellParamHostCollection = PortableHostCollection<HGCalMappingCellParamSoA>;

}  // namespace hgcal

#endif  // CondFormats_HGCalObjects_interface_HGCalMappingParameterHostCollection_h
