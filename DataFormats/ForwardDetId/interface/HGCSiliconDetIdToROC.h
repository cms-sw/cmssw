#ifndef DataFormats_ForwardDetId_HGCSiliconDetIdToROC_H
#define DataFormats_ForwardDetId_HGCSiliconDetIdToROC_H 1

#include "DataFormats/ForwardDetId/interface/HFNoseDetId.h"
#include "DataFormats/ForwardDetId/interface/HFNoseTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCalTriggerDetId.h"
#include "DataFormats/ForwardDetId/interface/HGCSiliconDetId.h"
#include <iostream>
#include <map>
#include <utility>
#include <vector>
#include <functional>

class HGCSiliconDetIdToROC {
public:
  /** This translated TriggerDetId to ROC and viceversa for HGCSilicon*/
  HGCSiliconDetIdToROC();

  int getROCNumber(HGCalTriggerDetId const& id) const {
    return getROCNumber(id.triggerCellU(), id.triggerCellV(), id.type());
  }
  int getROCNumber(HGCSiliconDetId const& id) const {
    return getROCNumber(id.triggerCellU(), id.triggerCellV(), id.type());
  }
  int getROCNumber(HFNoseDetId const& id) const {
    return getROCNumber(id.triggerCellU(), id.triggerCellV(), id.type());
  }
  int getROCNumber(HFNoseTriggerDetId const& id) const {
    return getROCNumber(id.triggerCellU(), id.triggerCellV(), id.type());
  }
  int getROCNumber(int triggerCellU, int triggerCellV, int type) const;
  std::vector<std::pair<int, int> > getTriggerId(int roc, int type) const;
  void print() const;

private:
  std::map<std::pair<int, int>, int> triggerIdToROC_;
  std::map<int, std::vector<std::pair<int, int> > > triggerIdFromROC_;
};
#endif
