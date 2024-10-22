#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
//
#include <initializer_list>
#include <vector>
#include <map>

namespace cond {

  namespace time {
    static const std::pair<const char*, TimeType> s_timeTypeMap[] = {std::make_pair("Run", cond::runnumber),
                                                                     std::make_pair("Time", cond::timestamp),
                                                                     std::make_pair("Lumi", cond::lumiid),
                                                                     std::make_pair("Hash", cond::hash),
                                                                     std::make_pair("User", cond::userid)};
    std::string timeTypeName(TimeType type) {
      if (type == invalid)
        return "";
      return s_timeTypeMap[type].first;
    }

    TimeType timeTypeFromName(const std::string& name) {
      for (auto const& i : s_timeTypeMap)
        if (name == i.first)
          return i.second;
      const cond::TimeTypeSpecs& theSpec = cond::findSpecs(name);
      return theSpec.type;
      //throwException( "TimeType \""+name+"\" is unknown.","timeTypeFromName");
    }

    Time_t tillTimeFromNextSince(Time_t nextSince, TimeType timeType) {
      if (nextSince == time::MAX_VAL)
        return time::MAX_VAL;
      if (timeType != (TimeType)TIMESTAMP) {
        return nextSince - 1;
      } else {
        auto unpackedTime = unpack(nextSince);
        //number of seconds in nanoseconds (avoid multiply and divide by 1e09)
        Time_t totalSecondsInNanoseconds = ((Time_t)unpackedTime.first) * 1000000000;
        //total number of nanoseconds
        Time_t totalNanoseconds = totalSecondsInNanoseconds + ((Time_t)(unpackedTime.second));
        //now decrementing of 1 nanosecond
        totalNanoseconds--;
        //now repacking (just change the value of the previous pair)
        unpackedTime.first = (unsigned int)(totalNanoseconds / 1000000000);
        unpackedTime.second = (unsigned int)(totalNanoseconds - (Time_t)unpackedTime.first * 1000000000);
        return pack(unpackedTime);
      }
    }

    Time_t tillTimeForIOV(Time_t since, unsigned int iovSize, TimeType timeType) {
      if (since == time::MAX_VAL)
        return time::MAX_VAL;
      if (timeType != (TimeType)TIMESTAMP) {
        return since + iovSize - 1;
      } else {
        auto unpackedTime = unpack(since);
        unpackedTime.first = unpackedTime.first + iovSize;
        return tillTimeFromNextSince(pack(unpackedTime), timeType);
      }
    }

    Time_t lumiTime(unsigned int run, unsigned int lumiId) { return cond::time::pack(std::make_pair(run, lumiId)); }

    Time_t sinceGroupSize(TimeType tp) {
      if (tp == TIMESTAMP)
        return SINCE_TIME_GROUP_SIZE;
      if (tp == LUMIID)
        return SINCE_LUMI_GROUP_SIZE;
      if (tp == HASH)
        return SINCE_HASH_GROUP_SIZE;
      return SINCE_RUN_GROUP_SIZE;
    }

    // framework conversions
    edm::IOVSyncValue toIOVSyncValue(Time_t time, TimeType timetype, bool startOrStop) {
      switch (timetype) {
        case RUNNUMBER:
          return edm::IOVSyncValue(edm::EventID(time,
                                                startOrStop ? 0 : edm::EventID::maxEventNumber(),
                                                startOrStop ? 0 : edm::EventID::maxEventNumber()));
        case LUMIID: {
          edm::LuminosityBlockID l(time);
          return edm::IOVSyncValue(
              edm::EventID(l.run(), l.luminosityBlock(), startOrStop ? 0 : edm::EventID::maxEventNumber()));
        }
        case TIMESTAMP:
          return edm::IOVSyncValue(edm::Timestamp(time));
        default:
          return edm::IOVSyncValue::invalidIOVSyncValue();
      }
    }

    Time_t fromIOVSyncValue(edm::IOVSyncValue const& time, TimeType timetype) {
      switch (timetype) {
        case RUNNUMBER:
          return time.eventID().run();
        case LUMIID: {
          edm::LuminosityBlockID lum(time.eventID().run(), time.luminosityBlockNumber());
          return lum.value();
        }
        case TIMESTAMP:
          return time.time().value();
        default:
          return 0;
      }
    }

    // the minimal maximum-time an IOV can extend to
    edm::IOVSyncValue limitedIOVSyncValue(Time_t time, TimeType timetype) {
      switch (timetype) {
        case RUNNUMBER:
          // last lumi and event of this run
          return edm::IOVSyncValue(edm::EventID(time, edm::EventID::maxEventNumber(), edm::EventID::maxEventNumber()));
        case LUMIID: {
          // the same lumiblock
          edm::LuminosityBlockID l(time);
          return edm::IOVSyncValue(edm::EventID(l.run(), l.luminosityBlock(), edm::EventID::maxEventNumber()));
        }
        case TIMESTAMP:
          // next event ?
          return edm::IOVSyncValue::invalidIOVSyncValue();
        default:
          return edm::IOVSyncValue::invalidIOVSyncValue();
      }
    }

    edm::IOVSyncValue limitedIOVSyncValue(edm::IOVSyncValue const& time, TimeType timetype) {
      switch (timetype) {
        case RUNNUMBER:
          // last event of this run
          return edm::IOVSyncValue(
              edm::EventID(time.eventID().run(), edm::EventID::maxEventNumber(), edm::EventID::maxEventNumber()));
        case LUMIID:
          // the same lumiblock
          return edm::IOVSyncValue(
              edm::EventID(time.eventID().run(), time.luminosityBlockNumber(), edm::EventID::maxEventNumber()));
        case TIMESTAMP:
          // same lumiblock
          return edm::IOVSyncValue(
              edm::EventID(time.eventID().run(), time.luminosityBlockNumber(), edm::EventID::maxEventNumber()));
        default:
          return edm::IOVSyncValue::invalidIOVSyncValue();
      }
    }

    std::string transactionIdForLumiTime(Time_t time, unsigned int iovSize, const std::string& secretKey) {
      auto unpackedTime = cond::time::unpack(time);
      unsigned int offset = 1 + iovSize;
      cond::Time_t id = 0;
      if (unpackedTime.second < offset) {
        id = lumiTime(unpackedTime.first, 1);
      } else {
        unsigned int res = (unpackedTime.second - offset) % iovSize;
        id = lumiTime(unpackedTime.first, unpackedTime.second - res);
      }
      std::stringstream transId;
      transId << id;
      if (!secretKey.empty()) {
        transId << "_" << secretKey;
      }
      return transId.str();
    }

  }  // namespace time

}  // namespace cond
