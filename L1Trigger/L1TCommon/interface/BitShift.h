#ifndef L1Trigger_L1TCommon_BitShift_h
#define L1Trigger_L1TCommon_BitShift_h

namespace l1t {

  inline int bitShift(int num, int bits) {
    if (num < 0) {
      return -1 * ((-1 * num) << bits);
    } else {
      return (num << bits);
    }
  }

}  // namespace l1t
#endif
