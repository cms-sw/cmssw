#ifndef DataFormats_ForwardDetId_HFNoseDetIdToModule_H
#define DataFormats_ForwardDetId_HFNoseDetIdToModule_H 1

#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HFNoseTriggerDetId.h"
#include <vector>

class HFNoseDetIdToModule {
public:
  /** This translated TriggerDetId to Module and viceversa for HFNose*/
  HFNoseDetIdToModule();

  static const HFNoseDetId getModule(HFNoseDetId const& id) {
    return id.geometryCell();
  }
  static const HFNoseDetId getModule(HFNoseTriggerDetId const& id) {
    return id.geometryCell();
  }
  std::vector<HFNoseDetId> getDetIds(HFNoseDetId const& id) const;
};
#endif
