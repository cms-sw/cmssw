#ifndef EVENTFILTER_PROCESSOR_PROCUTILS_H
#define EVENTFILTER_PROCESSOR_PROCUTILS_H

#include <string>

namespace evf{
  namespace utils{
    void procStat(std::ostringstream *);
    void uptime(std::ostringstream *);
    void mDiv(std::ostringstream *out, std::string name);
    void cDiv(std::ostringstream *out);
    void mDiv(std::ostringstream *out, std::string name, std::string value);
    void mDiv(std::ostringstream *out, std::string name, unsigned int value);
  }
}
#endif
