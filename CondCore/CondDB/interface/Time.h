#ifndef CondCore_CondDB_Time_h
#define CondCore_CondDB_Time_h
//
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"
//
#include <string>
#include <limits>

// imported from CondFormats/Common
namespace cond {  
  
  namespace time {

    // Time_t
    typedef cond::Time_t Time_t;

    const Time_t MAX(std::numeric_limits<Time_t>::max());

    const Time_t MIN(0);
  
    typedef cond::UnpackedTime UnpackedTime;

    typedef cond::TimeType TimeType;
  
    // TimeType
    typedef enum { INVALID=cond::invalid, RUNNUMBER=cond::runnumber, TIMESTAMP=cond::timestamp, LUMIID=cond::lumiid, HASH=cond::hash, USERID=cond::userid } _timetype;
  
    std::string timeTypeName(TimeType type);

    TimeType timeTypeFromName( const std::string& name );

    // constant defininig the (maximum) size of the iov groups 
    static constexpr unsigned int SINCE_GROUP_SIZE = 1000;

    Time_t tillTimeFromNextSince( Time_t nextSince, TimeType timeType );

  }  
  
}
#endif
  
