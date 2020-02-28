#include "L1Trigger/CSCTriggerPrimitives/interface/LCTContainer.h"

LCTContainer::LCTContainer(unsigned int trig_window_size) : match_trig_window_size_(trig_window_size) {}

CSCCorrelatedLCTDigi& LCTContainer::operator()(int bx, int match_bx, int lct) { return data[bx][match_bx][lct]; }

void LCTContainer::getTimeMatched(const int bx, std::vector<CSCCorrelatedLCTDigi>& lcts) const {
  for (unsigned int mbx = 0; mbx < match_trig_window_size_; mbx++) {
    for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
      // consider only valid LCTs
      if (not data[bx][mbx][i].isValid())
        continue;

      // remove duplicated LCTs
      if (std::find(lcts.begin(), lcts.end(), data[bx][mbx][i]) != lcts.end())
        continue;

      lcts.push_back(data[bx][mbx][i]);
    }
  }
}

void LCTContainer::getMatched(std::vector<CSCCorrelatedLCTDigi>& lcts) const {
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    std::vector<CSCCorrelatedLCTDigi> temp_lcts;
    LCTContainer::getTimeMatched(bx, temp_lcts);
    lcts.insert(std::end(lcts), std::begin(temp_lcts), std::end(temp_lcts));
  }
}

void LCTContainer::clear() {
  // Loop over all time windows
  for (int bx = 0; bx < CSCConstants::MAX_LCT_TBINS; bx++) {
    // Loop over all matched trigger windows
    for (unsigned int mbx = 0; mbx < match_trig_window_size_; mbx++) {
      // Loop over all stubs
      for (int i = 0; i < CSCConstants::MAX_LCTS_PER_CSC; i++) {
        data[bx][mbx][i].clear();
      }
    }
  }
}
