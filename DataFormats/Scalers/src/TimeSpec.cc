
#include "DataFormats/Scalers/interface/TimeSpec.h"

timespec TimeSpec::get_timespec() const {
  timespec ts;
  ts.tv_sec = tv_sec_;
  ts.tv_nsec = tv_nsec_;
  return ts;
}
