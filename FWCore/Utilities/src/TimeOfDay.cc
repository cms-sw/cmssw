#include "FWCore/Utilities/interface/TimeOfDay.h"
#include <iomanip>
#include <locale>
#include <ostream>
#include <time.h>

namespace {
  int const power[] = {1000*1000, 100*1000, 10*1000, 1000, 100, 10, 1};
}

namespace edm {

  TimeOfDay::TimeOfDay() : tv_(TimeOfDay::setTime_()) {
  }

  TimeOfDay::TimeOfDay(struct timeval const& tv) : tv_(tv) {
  }

  timeval
  TimeOfDay::setTime_() {
    timeval tv;
    gettimeofday(&tv, 0);
    return tv;
  }

  std::ostream&
  operator<<(std::ostream& os, TimeOfDay const& tod) {
    std::ios::fmtflags oldflags = os.flags(); // Save stream formats so they can be left unchanged.
    struct tm timebuf;
    localtime_r(&tod.tv_.tv_sec, &timebuf);
    typedef std::ostreambuf_iterator<char, std::char_traits<char> > Iter;
    std::time_put<char, Iter> const& tp = std::use_facet<std::time_put<char, Iter> >(std::locale());
    int precision = os.precision();
    Iter begin(os);
    if(precision == 0) {
      char const pattern[] = "%d-%b-%Y %H:%M:%S %Z";
      tp.put(begin, os, ' ', &timebuf, pattern, pattern + sizeof(pattern) - 1);
    } else {
      char const pattern[] = "%d-%b-%Y %H:%M:%S.";
      tp.put(begin, os, ' ', &timebuf, pattern, pattern + sizeof(pattern) - 1);
      precision = std::min(precision, 6);
      os << std::setfill('0') << std::setw(precision) << tod.tv_.tv_usec/power[precision] << ' ';
      tp.put(begin, os, ' ', &timebuf, 'Z');
    }
    os.flags(oldflags);
    return os;
  }
}
