#include "CondFormats/Common/interface/Time.h"

#include "FWCore/Utilities/interface/Exception.h"

namespace cond {

  const cond::TimeType timeTypeList[TIMETYPE_LIST_MAX] = {runnumber, timestamp, lumiid, hash, userid};

  const cond::TimeType timeTypeValues[] = {runnumber, timestamp, lumiid, hash, userid};

  std::string const& timeTypeNames(int i) {
    static const std::string local[] = {"runnumber", "timestamp", "lumiid", "hash", "userid"};
    return local[i];
  }

  const TimeTypeSpecs timeTypeSpecs[] = {TimeTypeTraits<runnumber>::specs(),
                                         TimeTypeTraits<timestamp>::specs(),
                                         TimeTypeTraits<lumiid>::specs(),
                                         TimeTypeTraits<hash>::specs(),
                                         TimeTypeTraits<userid>::specs()};

  // find spec by name
  const TimeTypeSpecs& findSpecs(std::string const& name) {
    size_t i = 0;
    for (; i < TIMETYPE_LIST_MAX; i++)
      if (name == timeTypeSpecs[i].name)
        return timeTypeSpecs[i];
    throw cms::Exception("invalid timetype: " + name);
    return timeTypeSpecs[0];  // compiler happy
  }

}  // namespace cond
