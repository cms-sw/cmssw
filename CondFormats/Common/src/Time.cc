#include "CondFormats/Common/interface/Time.h"


#include "CondCore/DBCommon/interface/Exception.h"

namespace cond{


 const TimeTypeSpecs timeTypeSpecs[] = {
    TimeTypeTraits<runnumber>::specs(),
    TimeTypeTraits<timestamp>::specs(),
    TimeTypeTraits<lumiid>::specs(),
    TimeTypeTraits<userid>::specs(),
  };


  
  // find spec by name
  const TimeTypeSpecs & findSpecs(std::string const & name) {
    size_t i=0;
    for (; i<TIMETYPE_LIST_MAX; i++)
      if (name==timeTypeSpecs[i].name) return timeTypeSpecs[i];
    throw cond::Exception("invalid timetype: "+name);
    return timeTypeSpecs[0]; // compiler happy
  }

}
