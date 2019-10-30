#include "DataFormats/ForwardDetId/interface/HFNoseDetIdToModule.h"

HFNoseDetIdToModule::HFNoseDetIdToModule() {}

std::vector<HFNoseDetId> HFNoseDetIdToModule::getDetIds(HFNoseDetId const& id) const {
  std::vector<HFNoseDetId> ids;
  int nCells = (id.type() == 0) ? HFNoseDetId::HFNoseFineN : HFNoseDetId::HFNoseCoarseN;
  for (int u = 0; u < 2 * nCells; ++u) {
    for (int v = 0; v < 2 * nCells; ++v) {
      if (((v - u) < nCells) && (u - v) <= nCells) {
        HFNoseDetId newId(id.zside(), id.type(), id.layer(), id.waferU(), id.waferV(), u, v);
        ids.emplace_back(newId);
      }
    }
  }
  return ids;
}

std::vector<HFNoseTriggerDetId> HFNoseDetIdToModule::getTriggerDetIds(HFNoseDetId const& id) const {
  std::vector<HFNoseTriggerDetId> ids;
  int nCells = HFNoseDetId::HFNoseFineN / HFNoseDetId::HFNoseFineTrigger;
  for (int u = 0; u < 2 * nCells; ++u) {
    for (int v = 0; v < 2 * nCells; ++v) {
      if (((v - u) < nCells) && (u - v) <= nCells) {
        HFNoseTriggerDetId newId(HFNoseTrigger, id.zside(), id.type(), id.layer(), id.waferU(), id.waferV(), u, v);
        ids.emplace_back(newId);
      }
    }
  }
  return ids;
}
