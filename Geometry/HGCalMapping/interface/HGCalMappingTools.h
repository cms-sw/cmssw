#ifndef _geometry_hgcalmapping_hgcalmappingtools_h_
#define _geometry_hgcalmapping_hgcalmappingtools_h_

#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCScintillatorDetId.h"
#include "DataFormats/HGCalDigi/interface/HGCalElectronicsId.h"

namespace hgcal {

  namespace mappingtools {

    uint16_t getEcondErx(uint16_t chip, uint16_t half);
    uint32_t getElectronicsId(
        bool zside, uint16_t fedid, uint16_t captureblock, uint16_t econdidx, int cellchip, int cellhalf, int cellseq);
    uint32_t getSiDetId(bool zside, int moduleplane, int moduleu, int modulev, int celltype, int celliu, int celliv);
    uint32_t getSiPMDetId(bool zside, int moduleplane, int modulev, int celltype, int celliu, int celliv);
  }  // namespace mappingtools
}  // namespace hgcal

#endif
