#ifndef CondFormats_HGCalObjects_interface_HGCalMappingParameterSoA_h
#define CondFormats_HGCalObjects_interface_HGCalMappingParameterSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

#include "CondFormats/HGCalObjects/interface/HGCalMappingModuleIndexer.h"
#include "CondFormats/HGCalObjects/interface/HGCalMappingCellIndexer.h"

namespace hgcal {

  // Generate structure of module-level (ECON-D) arrays (SoA) layout with module mapping information
  GENERATE_SOA_LAYOUT(HGCalMappingModuleParamSoALayout,
                      SOA_COLUMN(bool, valid),
                      SOA_COLUMN(bool, zside),
                      SOA_COLUMN(bool, isSiPM),
                      SOA_COLUMN(int, plane),
                      SOA_COLUMN(int, i1),
                      SOA_COLUMN(int, i2),
                      SOA_COLUMN(uint8_t, irot),
                      SOA_COLUMN(int8_t, celltype),
                      SOA_COLUMN(uint16_t, typeidx),
                      SOA_COLUMN(uint16_t, fedid),
                      SOA_COLUMN(uint16_t, slinkidx),
                      SOA_COLUMN(uint16_t, captureblock),
                      SOA_COLUMN(uint16_t, econdidx),
                      SOA_COLUMN(uint16_t, captureblockidx),
                      SOA_COLUMN(uint32_t, eleid),
                      SOA_COLUMN(uint32_t, detid),
                      SOA_COLUMN(uint32_t, cassette))
  using HGCalMappingModuleParamSoA = HGCalMappingModuleParamSoALayout<>;

  // Generate structure of module-level (ECON-T) arrays (SoA) layout with module mapping information
  GENERATE_SOA_LAYOUT(HGCalMappingModuleTriggerParamSoALayout,
                      SOA_COLUMN(bool, valid),
                      SOA_COLUMN(bool, zside),
                      SOA_COLUMN(bool, isSiPM),
                      SOA_COLUMN(int, plane),
                      SOA_COLUMN(int, i1),
                      SOA_COLUMN(int, i2),
                      SOA_COLUMN(uint8_t, irot),
                      SOA_COLUMN(int, celltype),
                      SOA_COLUMN(uint16_t, typeidx),
                      SOA_COLUMN(uint16_t, fedid),
                      SOA_COLUMN(uint16_t, slinkidx),
                      SOA_COLUMN(uint16_t, econtidx),
                      SOA_COLUMN(uint32_t, muxid),
                      SOA_COLUMN(uint32_t, trigdetid),
                      SOA_COLUMN(uint32_t, cassette))
  using HGCalMappingModuleTriggerParamSoA = HGCalMappingModuleTriggerParamSoALayout<>;

  // Generate structure of channel-level arrays (SoA) layout with cell mapping information for both silicon and SiPM
  GENERATE_SOA_LAYOUT(HGCalMappingCellParamSoALayout,
                      SOA_COLUMN(bool, valid),
                      SOA_COLUMN(bool, isHD),
                      SOA_COLUMN(bool, iscalib),
                      SOA_COLUMN(int, caliboffset),
                      SOA_COLUMN(bool, isSiPM),
                      SOA_COLUMN(uint16_t, typeidx),
                      SOA_COLUMN(uint16_t, chip),
                      SOA_COLUMN(uint16_t, half),
                      SOA_COLUMN(uint16_t, seq),
                      SOA_COLUMN(uint16_t, rocpin),
                      SOA_COLUMN(int, sensorcell),
                      SOA_COLUMN(int, triglink),
                      SOA_COLUMN(int, trigcell),
                      SOA_COLUMN(int, i1),  // iu/iring
                      SOA_COLUMN(int, i2),  // iv/iphi
                      SOA_COLUMN(int, t),
                      SOA_COLUMN(float, trace),
                      SOA_COLUMN(uint32_t, eleid),
                      SOA_COLUMN(uint32_t, detid))
  using HGCalMappingCellParamSoA = HGCalMappingCellParamSoALayout<>;

  // Generate structure of channel-level arrays (SoA) layout with module mapping information
  GENERATE_SOA_LAYOUT(HGCalDenseIndexInfoSoALayout,
                      SOA_COLUMN(uint32_t, fedId),
                      SOA_COLUMN(uint32_t, fedReadoutSeq),
                      SOA_COLUMN(uint32_t, detid),
                      SOA_COLUMN(uint32_t, eleid),
                      SOA_COLUMN(uint32_t, modInfoIdx),
                      SOA_COLUMN(uint32_t, cellInfoIdx),
                      SOA_COLUMN(uint32_t, chNumber),
                      SOA_COLUMN(uint32_t, layer),
                      SOA_COLUMN(float, eta),
                      SOA_COLUMN(float, phi),
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z))
  using HGCalDenseIndexInfoSoA = HGCalDenseIndexInfoSoALayout<>;

  // Generatie structure of tirgger-cell level arrays (SoA) layout with module mapping info
  GENERATE_SOA_LAYOUT(HGCalDenseIndexTriggerInfoSoALayout,
                      SOA_COLUMN(uint32_t, fedId),
                      SOA_COLUMN(uint16_t, fedReadoutSeq),
                      SOA_COLUMN(uint32_t, trigdetid),
                      SOA_COLUMN(uint32_t, muxid),
                      SOA_COLUMN(uint32_t, modInfoIdx),
                      SOA_COLUMN(uint32_t, cellInfoIdx),
                      SOA_COLUMN(uint32_t, TCNumber),
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z))
  using HGCalDenseIndexTriggerInfoSoA = HGCalDenseIndexTriggerInfoSoALayout<>;

}  // namespace hgcal

#endif  // CondFormats_HGCalObjects_interface_HGCalMappingParameterSoA_h
