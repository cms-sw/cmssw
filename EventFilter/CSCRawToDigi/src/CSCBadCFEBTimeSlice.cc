#include "EventFilter/CSCRawToDigi/interface/CSCBadCFEBTimeSlice.h"
#include <cassert>

const CSCBadCFEBWord& CSCBadCFEBTimeSlice::word(int i) const {
  assert(i >= 0 && i < 4);
  return theWords[i];
}

bool CSCBadCFEBTimeSlice::check() const {
  // demand all four words check out
  bool result = true;
  for (auto theWord : theWords) {
    result &= theWord.check();
  }
  return result;
}
