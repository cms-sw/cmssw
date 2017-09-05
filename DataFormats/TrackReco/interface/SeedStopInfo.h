#ifndef DataFormats_TrackReco_SeedStopInfo_h
#define DataFormats_TrackReco_SeedStopInfo_h

#include "DataFormats/TrackReco/interface/SeedStopReason.h"

class SeedStopInfo {
public:
  SeedStopInfo() {}
  ~SeedStopInfo() = default;

  void setStopReason(SeedStopReason value) { stopReason_ = value; }
  SeedStopReason stopReason() const { return stopReason_; }
  unsigned char stopReasonUC() const { return static_cast<unsigned char>(stopReason_); }

private:
  SeedStopReason stopReason_ = SeedStopReason::UNINITIALIZED;
};

#endif
