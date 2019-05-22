#ifndef XDAQ_COMPAT
#define XDAQ_COMPAT

#ifndef HAVE_XDAQ
#include <string>   // std::string
#include <cstdarg>  // va_list, va_start
#include <ostream>  // std::ostream

/* Replace log4cplus::Logger */
namespace log4cplus {
  typedef std::ostream* Logger;
}

namespace toolbox {
  std::string toString(const char* format, ...);
}

#endif  // HAVE_XDAQ

#endif  // XDAQ_COMPAT
