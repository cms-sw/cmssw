#ifndef __HLX_TIMESTAMP_H__
#define __HLX_TIMESTAMP_H__

#include <string>

namespace HCAL_HLX{
  
  class TimeStamp{
  public:
    std::string TimeStampLong();
    std::string TimeStampYYYYMMDD();
    std::string TimeStampYYYYMM();    
  };
  
}

#endif
