#ifndef FWCore_Utilities_TimeOfDay_h
#define FWCore_Utilities_TimeOfDay_h

#include <sys/time.h>
#include <chrono>
#include <iosfwd>

namespace edm {
  struct TimeOfDay {
    TimeOfDay();
    explicit TimeOfDay(struct timeval const& tv);
    explicit TimeOfDay(std::chrono::system_clock::time_point const& tp);

    struct timeval tv_;

  private:
    static struct timeval setTime_();
  };

  std::ostream& operator<<(std::ostream& os, TimeOfDay const& tod);
}  // namespace edm

#endif
