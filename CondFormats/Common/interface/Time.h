#ifndef CondCommon_Time_h
#define CondCommon_Time_h
#include <utility>
#include <string>
#include <limits>
// #include <boost/cstdint.hpp>

#include "CondFormats/Common/interface/hash64.h"

namespace cond {

  // typedef uint64_t  Time_t;
  typedef unsigned long long uint64_t;  // avoid typedef to long on 64 bit
  typedef unsigned long long Time_t;
  typedef std::pair<unsigned int, unsigned int> UnpackedTime;

  typedef std::pair<Time_t, Time_t> ValidityInterval;

  typedef enum { invalid = -1, runnumber = 0, timestamp, lumiid, hash, userid } TimeType;
  const unsigned int TIMETYPE_LIST_MAX = 5;

  extern const cond::TimeType timeTypeList[TIMETYPE_LIST_MAX];

  extern const cond::TimeType timeTypeValues[];

  std::string const& timeTypeNames(int);

  const Time_t TIMELIMIT(std::numeric_limits<Time_t>::max());

  const Time_t invalidTime(0);

  template <TimeType type>
  struct RealTimeType {};

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

  template <>
  struct RealTimeType<runnumber> {
    // 0, run number
    typedef unsigned int type;
  };

  template <>
  struct RealTimeType<timestamp> {
    // sec, nanosec
    typedef uint64_t type;
  };

  template <>
  struct RealTimeType<lumiid> {
    // run, lumi-seg
    typedef uint64_t type;
  };

  template <>
  struct RealTimeType<hash> {
    typedef uint64_t type;
  };

  template <>
  struct RealTimeType<userid> {
    typedef uint64_t type;
  };

  template <TimeType type>
  struct TimeTypeTraits {
    static const TimeTypeSpecs& specs() {
      static const TimeTypeSpecs local = {type,
                                          timeTypeNames(type),
                                          1,
                                          std::numeric_limits<typename RealTimeType<type>::type>::max(),
                                          cond::invalidTime};
      return local;
    }
  };

  extern const TimeTypeSpecs timeTypeSpecs[];

  // find spec by name
  const TimeTypeSpecs& findSpecs(std::string const& name);

}  // namespace cond
#endif
