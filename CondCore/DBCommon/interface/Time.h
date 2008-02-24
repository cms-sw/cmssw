#ifndef DBCommon_Time_h
#define DBCommon_Time_h 
#include<utility>
namespace cond{  
  //typedef unsigned int Time_t;
  typedef unsigned long long Time_t;
  typedef std::pair<Time_t, Time_t> ValidityInterval;
  typedef enum { runnumber=0,timestamp,lumiid } TimeType;
  const unsigned int TIMETYPE_LIST_MAX=3;
  const cond::TimeType TimeTypeList[TIMETYPE_LIST_MAX]={runnumber,timestamp,lumiid};
  static const Time_t TIMELIMIT(0xFFFFFFFF);
}
#endif
