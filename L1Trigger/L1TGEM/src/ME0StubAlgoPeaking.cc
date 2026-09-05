#include "L1Trigger/L1TGEM/interface/ME0StubAlgoPeaking.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

std::vector<ME0StubPrimitive> l1t::me0::PeakingManager::processSegments(const int partition,
                                                                        const std::vector<ME0StubPrimitive>& newSegs) {
  auto trig = trigger_[partition];
  auto oldSegs = segs_[0][partition];
  auto oldestSegs = segs_[1][partition];

  std::vector<ME0StubPrimitive> output;

  for (size_t i = 0; i < trig.size(); ++i) {
    if (trig[i]) {
      output.push_back(oldSegs[i]);
      trigger_[partition][i] = false;  // reset trigger after firing
    } else if (oldSegs[i].layerCount() > 0 && oldestSegs[i].layerCount() <= 0) {
      if (newSegs[i].layerCount() == 0) {
        output.push_back(
            oldSegs[i]);  // trigger stays false as old segment is still existed while new segment is not existed
      } else {
        output.push_back(ME0StubPrimitive(
            0, 0, 0, i, partition));  // output is empty segment as new segment is existed while old segment is existed
        trigger_[partition][i] =
            true;  // set trigger as new segment and old segment are existed while oldest segment is not existed
      }
    } else {
      output.push_back(ME0StubPrimitive(
          0, 0, 0, i, partition));  // output is empty segment as new segment is existed while old segment is existed
    }
  }

  // Update segs
  segs_[1][partition] = oldSegs;
  segs_[0][partition] = newSegs;

  return output;
}
