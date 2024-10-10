#ifndef CondFormats_HGCalObjects_interface_HGCalMappingParameterHost_h
#define CondFormats_HGCalObjects_interface_HGCalMappingParameterHost_h

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingParameterSoA.h"

namespace hgcal {

  // SoA with channel-level module mapping parameters in host memory:
  using HGCalMappingModuleParamHost = PortableHostCollection<HGCalMappingModuleParamSoA>;

  // SoA with channel-level cell mapping parameters in host memory for both Si and SiPM channels:
  using HGCalMappingCellParamHost = PortableHostCollection<HGCalMappingCellParamSoA>;

  //SoA with detailed indices corresponding to the dense index in use
  using HGCalDenseIndexInfoHost = PortableHostCollection<HGCalDenseIndexInfoSoA>;

}  // namespace hgcal

#endif  // CondFormats_HGCalObjects_interface_HGCalMappingParameterHost_h
