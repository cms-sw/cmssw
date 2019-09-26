#include "DataFormats/ForwardDetId/interface/HGCSiliconDetIdToModule.h"

HGCSiliconDetIdToModule::HGCSiliconDetIdToModule() {}

std::vector<HGCSiliconDetId> HGCSiliconDetIdToModule::getDetIds(HGCSiliconDetId const& id) const {
  std::vector<HGCSiliconDetId> ids;
  int nCells = (id.type() == 0) ? HGCSiliconDetId::HGCalFineN : HGCSiliconDetId::HGCalCoarseN;
  for (int u = 0; u < 2 * nCells; ++u) {
    for (int v = 0; v < 2 * nCells; ++v) {
      if (((v - u) < nCells) && (u - v) <= nCells) {
        HGCSiliconDetId newId(id.det(), id.zside(), id.type(), id.layer(), id.waferU(), id.waferV(), u, v);
        ids.emplace_back(newId);
      }
    }
  }
  return ids;
}
