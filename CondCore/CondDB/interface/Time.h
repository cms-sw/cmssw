#ifndef CondCore_CondDB_Time_h
#define CondCore_CondDB_Time_h
//
#include "CondFormats/Common/interface/Time.h"
#include "CondFormats/Common/interface/TimeConversions.h"
#include "FWCore/Framework/interface/IOVSyncValue.h"
//
#include <string>
#include <limits>
#include <type_traits>

// imported from CondFormats/Common
namespace cond {

  namespace time {

    // Time_t
    typedef cond::Time_t Time_t;

    const Time_t MAX_VAL(std::numeric_limits<Time_t>::max());

    const Time_t MIN_VAL(0);

    const unsigned int SECONDS_PER_LUMI(23);

    static constexpr const char* const MAX_TIMESTAMP = "9999-12-31 23:59:59.000";

    typedef cond::UnpackedTime UnpackedTime;

    typedef cond::TimeType TimeType;

    // TimeType
    static constexpr TimeType INVALID = cond::invalid;
    static constexpr TimeType RUNNUMBER = cond::runnumber;
    static constexpr TimeType TIMESTAMP = cond::timestamp;
    static constexpr TimeType LUMIID = cond::lumiid;
    static constexpr TimeType HASH = cond::hash;
    static constexpr TimeType USERID = cond::userid;

    std::string timeTypeName(TimeType type);

    TimeType timeTypeFromName(const std::string& name);

    // constant defininig the (maximum) size of the iov groups
    static constexpr unsigned int SINCE_RUN_GROUP_SIZE = 1000;
    // 36000 << 32 ( corresponding to 10h )
    static constexpr unsigned long SINCE_TIME_GROUP_SIZE = 154618822656000;
    static constexpr unsigned int SINCE_LUMI_GROUP_SIZE = SINCE_RUN_GROUP_SIZE;
    static constexpr unsigned int SINCE_HASH_GROUP_SIZE = SINCE_RUN_GROUP_SIZE;

    Time_t sinceGroupSize(TimeType tp);

    Time_t tillTimeFromNextSince(Time_t nextSince, TimeType timeType);

    Time_t lumiTime(unsigned int run, unsigned int lumiId);

    // conversion from framework types
    edm::IOVSyncValue toIOVSyncValue(cond::Time_t time, TimeType timetype, bool startOrStop);

    Time_t fromIOVSyncValue(edm::IOVSyncValue const& time, TimeType timetype);

    // min max sync value....
    edm::IOVSyncValue limitedIOVSyncValue(Time_t time, TimeType timetype);

    edm::IOVSyncValue limitedIOVSyncValue(edm::IOVSyncValue const& time, TimeType timetype);

  }  // namespace time

}  // namespace cond
#endif
