#ifndef EventFilter_CTPPSRawToDigi_ElectronicIndex_H
#define EventFilter_CTPPSRawToDigi_ElectronicIndex_H

namespace pps::pixel {
  struct ElectronicIndex {
    int link;
    int roc;
    int dcol;
    int pxid;
  };
}  // namespace pps::pixel

#endif
