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

std::vector<HGCalTriggerDetId> HGCSiliconDetIdToModule::getDetTriggerIds(HGCSiliconDetId const& id) const {
  std::vector<HGCalTriggerDetId> ids;
  int nCells = HGCSiliconDetId::HGCalFineN / HGCSiliconDetId::HGCalFineTrigger;
  int subdet = (id.det() == DetId::HGCalEE) ? HGCalEETrigger : HGCalHSiTrigger;
  for (int u = 0; u < 2 * nCells; ++u) {
    for (int v = 0; v < 2 * nCells; ++v) {
      if (((v - u) < nCells) && (u - v) <= nCells) {
        HGCalTriggerDetId newId(subdet, id.zside(), id.type(), id.layer(), id.waferU(), id.waferV(), u, v);
        ids.emplace_back(newId);
      }
    }
  }
  return ids;
}
