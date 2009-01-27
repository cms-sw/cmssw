#ifndef CondCommon_Time_h
#define CondCommon_Time_h 
#include<utility>
#include <string>
#include <limits>
#include <boost/cstdint.hpp>

// to be fixed
#include "CondCore/DBCommon/interface/Exception.h"


namespace cond{


  typedef uint64_t  Time_t;
  typedef std::pair<unsigned int, unsigned int> UnpackedTime;

  typedef std::pair<Time_t, Time_t> ValidityInterval;

  typedef enum { runnumber=0, timestamp, lumiid, userid } TimeType;
  const unsigned int TIMETYPE_LIST_MAX=4;

  const cond::TimeType timeTypeList[TIMETYPE_LIST_MAX]=
    {runnumber,timestamp,lumiid,userid};

  const cond::TimeType timeTypeValues[]=
    {runnumber,timestamp,lumiid,userid};

  const std::string timeTypeNames[]=
    {"runnumber","timestamp","lumiid","userid"};


  const Time_t TIMELIMIT(std::numeric_limits<Time_t>::max());

  template<TimeType type>
  struct RealTimeType {
  };

  
  struct TimeTypeSpecs {
    // the enum
    TimeType type;
    // the name
    std::string name;
    // begin, end, and invalid 
    Time_t beginValue;
    Time_t endValue;
    Time_t invalidValue;
    
  }; 

  
  template<> struct RealTimeType<runnumber> {
    // 0, run number
    typedef unsigned int type; 
  };

  template<> struct RealTimeType<timestamp> {
    // sec, nanosec
    typedef uint64_t  type; 
  };
  
 
  template<> struct RealTimeType<lumiid> {
    // run, lumi-seg
    typedef uint64_t  type; 
  };

  template<> struct RealTimeType<userid> {
    typedef uint64_t  type; 
  };

  
  template<TimeType type>
  struct TimeTypeTraits {
    static  const TimeTypeSpecs & specs() {
      static const TimeTypeSpecs local = { 
	type,
	timeTypeNames[type],
	1,
	std::numeric_limits<typename RealTimeType<type>::type>::max(),
	0
      };
      return local;
    }
  };

  const TimeTypeSpecs timeTypeSpecs[] = {
    TimeTypeTraits<runnumber>::specs(),
    TimeTypeTraits<timestamp>::specs(),
    TimeTypeTraits<lumiid>::specs(),
    TimeTypeTraits<userid>::specs(),
  };

  // find spec by name
  inline const TimeTypeSpecs & findSpecs(std::string const & name) {
    size_t i=0;
    for (; i<TIMETYPE_LIST_MAX; i++)
      if (name==timeTypeSpecs[i].name) return timeTypeSpecs[i];
    throw cond::Exception("invalid timetype: "+name);
    return timeTypeSpecs[0]; // compiler happy
  }

}

#endif
