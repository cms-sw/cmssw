#include "Geometry/HGCalMapping/interface/HGCalMappingTools.h"

namespace hgcal {

  namespace mappingtools {

    uint16_t getEcondErx(uint16_t chip, uint16_t half) { return chip * 2 + half; }

    uint32_t getElectronicsId(
        bool zside, uint16_t fedid, uint16_t captureblock, uint16_t econdidx, int cellchip, int cellhalf, int cellseq) {
      uint16_t econderx = getEcondErx(cellchip, cellhalf);

      return HGCalElectronicsId(zside, fedid, captureblock, econdidx, econderx, cellseq).raw();
    }

    uint32_t getSiDetId(bool zside, int moduleplane, int moduleu, int modulev, int celltype, int celliu, int celliv) {
      DetId::Detector det = moduleplane <= 26 ? DetId::Detector::HGCalEE : DetId::Detector::HGCalHSi;
      int zp(zside ? 1 : -1);

      return HGCSiliconDetId(det, zp, celltype, moduleplane, moduleu, modulev, celliu, celliv).rawId();
    }

    uint32_t getSiPMDetId(bool zside, int moduleplane, int modulev, int celltype, int celliu, int celliv) {
      int layer = moduleplane - 25;
      int type = 0;  // depends on SiPM size to be updated with new geometry

      int ring = (zside ? celliu : (-1) * celliu);
      int iphi = modulev * 8 + celliv + 1;

      return HGCScintillatorDetId(type, layer, ring, iphi, false, celltype).rawId();
    }
  }  // namespace mappingtools
}  // namespace hgcal
