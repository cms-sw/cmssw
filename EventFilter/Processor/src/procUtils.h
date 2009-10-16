#ifndef EVENTFILTER_PROCESSOR_PROCUTILS_H
#define EVENTFILTER_PROCESSOR_PROCUTILS_H

#include <string>


namespace evf{
  namespace utils{
    void procStat(std::ostringstream *);
    void uptime(std::ostringstream *);
  }
}
#endif
