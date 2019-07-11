#ifndef EventFilter_CTPPSRawToDigi_CTPPSElectronicIndex_H
#define EventFilter_CTPPSRawToDigi_CTPPSElectronicIndex_H

namespace ctppspixelobjects {
  struct ElectronicIndex {
    int link;
    int roc;
    int dcol;
    int pxid;
  };
}  // namespace ctppspixelobjects

#endif
