#ifndef CondFormats_HGCalObjects_interface_HGCalMappingParameterSoA_h
#define CondFormats_HGCalObjects_interface_HGCalMappingParameterSoA_h

#include "DataFormats/SoATemplate/interface/SoACommon.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "DataFormats/SoATemplate/interface/SoAView.h"

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
                      SOA_COLUMN(int, celltype),
                      SOA_COLUMN(uint16_t, typeidx),
                      SOA_COLUMN(uint16_t, fedid),
                      SOA_COLUMN(uint16_t, slinkidx),
                      SOA_COLUMN(uint16_t, captureblock),
                      SOA_COLUMN(uint16_t, econdidx),
                      SOA_COLUMN(uint16_t, captureblockidx),
                      SOA_COLUMN(uint32_t, eleid),
                      SOA_COLUMN(uint32_t, detid))
  using HGCalMappingModuleParamSoA = HGCalMappingModuleParamSoALayout<>;

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
                      SOA_COLUMN(float, x),
                      SOA_COLUMN(float, y),
                      SOA_COLUMN(float, z))
  using HGCalDenseIndexInfoSoA = HGCalDenseIndexInfoSoALayout<>;

}  // namespace hgcal

#endif  // CondFormats_HGCalObjects_interface_HGCalMappingParameterSoA_h
