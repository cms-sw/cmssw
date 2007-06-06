#include "xdata/UnsignedInteger32.h"
#include <string>


namespace evf
{
  class RunBase {
  public:
    static xdata::UnsignedInteger32 runNumber_;
    static std::string              sourceId_;
  };
  
} // namespace evf
