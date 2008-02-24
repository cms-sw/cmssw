#ifndef DBCommon_Time_h
#define DBCommon_Time_h 
#include<utility>
namespace cond{  
  //typedef unsigned int Time_t;
  typedef unsigned long long Time_t;
  typedef std::pair<Time_t, Time_t> ValidityInterval;
  enum TimeType{ runnumber=0,timestamp,lumiid };
  const unsigned int TIMETYPE_LIST_MAX=3;
  cond::TimeType TimeTypeList[TIMETYPE_LIST_MAX]={runnumber,timestamp,lumiid};
}
#endif
