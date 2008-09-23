#ifndef __HLX_TIMESTAMP_H__
#define __HLX_TIMESTAMP_H__

#include <string>
#include <ctime>

namespace HCAL_HLX{
  
  class TimeStamp{
  public:
    std::string TimeStampLong( time_t rawtime = 0 );
    std::string TimeStampYYYYMM( time_t  rawtime = 0);
    std::string TimeStampYYYYMMDD( time_t  rawtime = 0);
  };
}

#endif
