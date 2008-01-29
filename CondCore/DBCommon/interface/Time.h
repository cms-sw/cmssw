#ifndef DBCommon_Time_h
#define DBCommon_Time_h 
#include<utility>
namespace cond{  
  //typedef unsigned int Time_t;
  typedef unsigned long long Time_t;
  typedef std::pair<Time_t, Time_t> ValidityInterval;
  typedef enum { runnumber,timestamp } TimeType;
}
#endif
