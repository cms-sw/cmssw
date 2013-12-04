#ifndef CondCore_CondDB_Time_h
#define CondCore_CondDB_Time_h
//
//#include<utility>
#include <string>
#include <limits>

// imported from CondFormats/Common
namespace conddb {  
  
  // typedef uint64_t  Time_t;
  typedef unsigned long long uint64_t; // avoid typedef to long on 64 bit

  namespace time {

    // Time_t
    typedef unsigned long long Time_t;

    const Time_t MAX(std::numeric_limits<Time_t>::max());

    const Time_t MIN(0);
  
    typedef std::pair<unsigned int, unsigned int> UnpackedTime;
  
    // TimeType
    typedef enum { INVALID=-1, RUNNUMBER=0, TIMESTAMP, LUMIID, HASH, USERID } TimeType;
  
    std::string timeTypeName(TimeType type);

    TimeType timeTypeFromName( const std::string& name );

    // constant defininig the (maximum) size of the iov groups 
    static constexpr unsigned int SINCE_GROUP_SIZE = 1000;

  }  
  
  typedef time::Time_t Time_t;
  typedef time::UnpackedTime UnpackedTime;
  typedef time::TimeType TimeType;
  
}
#endif
  
