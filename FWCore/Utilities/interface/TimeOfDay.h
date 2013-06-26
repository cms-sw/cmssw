#ifndef FWCore_Utilities_TimeOfDay_h
#define FWCore_Utilities_TimeOfDay_h

#include <sys/time.h>
#include <iosfwd>

namespace edm {
  struct TimeOfDay {
    TimeOfDay();
    explicit TimeOfDay(struct timeval const& tv);

    struct timeval tv_;
  private:
    static struct timeval setTime_();
  };

  std::ostream&
  operator<<(std::ostream& os, TimeOfDay const& tod);
}

#endif
