#include <ctime>
#include <iomanip>
#include <locale>
#include <ostream>

#include "FWCore/Utilities/interface/TimeOfDay.h"

namespace {
  int const power[] = {1000 * 1000, 100 * 1000, 10 * 1000, 1000, 100, 10, 1};
}

namespace edm {

  TimeOfDay::TimeOfDay() : tv_(TimeOfDay::setTime_()) {}

  TimeOfDay::TimeOfDay(struct timeval const& tv) : tv_(tv) {}

  TimeOfDay::TimeOfDay(std::chrono::system_clock::time_point const& tp) {
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(tp.time_since_epoch()).count();
    tv_.tv_sec = us / 1000000;
    tv_.tv_usec = us % 1000000;
  }

  timeval TimeOfDay::setTime_() {
    timeval tv;
    gettimeofday(&tv, nullptr);
    return tv;
  }

  std::ostream& operator<<(std::ostream& os, TimeOfDay const& tod) {
    auto oldflags = os.flags();  // save the stream format flags so they can be restored
    auto oldfill = os.fill();    // save the stream fill character so it can be restored
    struct tm timebuf;
    localtime_r(&tod.tv_.tv_sec, &timebuf);
    typedef std::ostreambuf_iterator<char, std::char_traits<char> > Iter;
    std::time_put<char, Iter> const& tp = std::use_facet<std::time_put<char, Iter> >(std::locale());
    int precision = os.precision();
    Iter begin(os);
    if (precision == 0) {
      char const pattern[] = "%d-%b-%Y %H:%M:%S %Z";
      tp.put(begin, os, ' ', &timebuf, pattern, pattern + sizeof(pattern) - 1);
    } else {
      char const pattern[] = "%d-%b-%Y %H:%M:%S.";
      tp.put(begin, os, ' ', &timebuf, pattern, pattern + sizeof(pattern) - 1);
      precision = std::min(precision, 6);
      os << std::setfill('0') << std::setw(precision) << tod.tv_.tv_usec / power[precision] << ' ';
      tp.put(begin, os, ' ', &timebuf, 'Z');
    }
    os.flags(oldflags);
    os.fill(oldfill);
    return os;
  }

}  // namespace edm
