#ifndef DataFormats_ForwardDetId_HGCSiliconDetIdToModule_H
#define DataFormats_ForwardDetId_HGCSiliconDetIdToModule_H 1

#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include <vector>

class HGCSiliconDetIdToModule {
public:
  /** This translated TriggerDetId to Module and viceversa for HGCSilicon*/
  HGCSiliconDetIdToModule();

  static const HGCSiliconDetId getModule(HGCalTriggerDetId const& id) { return id.moduleId(); }
  static const HGCSiliconDetId getModule(HGCSiliconDetId const& id) { return id.moduleId(); }
  std::vector<HGCSiliconDetId> getDetIds(HGCSiliconDetId const& id) const;
  std::vector<HGCalTriggerDetId> getDetTriggerIds(HGCSiliconDetId const& id) const;
};
#endif
