#ifndef L1Trigger_TrackerDTC_StubDTC_h
#define L1Trigger_TrackerDTC_StubDTC_h

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"
#include "DataFormats/L1TrackTrigger/interface/TTBV.h"
#include "L1Trigger/TrackerDTC/interface/StubGL.h"

namespace trackerDTC {

  /*! \class  trackerDTC::StubDTC
   *  \brief  Class to represent an outer tracker Stub send from DTC
   *  \author Thomas Schuh
   *  \date   2025, Dec
   */
  class StubDTC {
  public:
    StubDTC(const StubGL&, int);
    ~StubDTC() = default;
    // underlying global stub
    const StubGL* stubGL() const { return stubGL_; }
    // no bit overflows
    bool valid() const { return valid_; }
    // bit accurate representation of Stub
    const tt::FrameStub& frame() const { return frame_; }

  private:
    // underlying global stub
    const StubGL* stubGL_;
    // no bit overflows
    bool valid_;
    // bit accurate representation of Stub
    tt::FrameStub frame_;
  };

}  // namespace trackerDTC

#endif
