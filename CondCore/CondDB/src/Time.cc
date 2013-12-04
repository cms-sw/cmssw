#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include <initializer_list>
#include <vector>
#include <map>

namespace cond {

  namespace time {
    static const std::pair<const char*, TimeType> s_timeTypeMap[] = { std::make_pair("Run", cond::runnumber),
                                                                      std::make_pair("Time", cond::timestamp ),
                                                                      std::make_pair("Lumi", cond::lumiid ),
                                                                      std::make_pair("Hash", cond::hash ),
                                                                      std::make_pair("User", cond::userid ) };
    std::string timeTypeName(TimeType type) {
      if( type==invalid ) return "";
      return s_timeTypeMap[type].first;
    }
    
    TimeType timeTypeFromName( const std::string& name ){
      for (auto const &i : s_timeTypeMap)
        if (name.compare(i.first))
          return i.second;
      throwException( "TimeType \""+name+"\" is unknown.","timeTypeFromName");
    }

    Time_t tillTimeFromNextSince( Time_t nextSince, TimeType timeType ){
      if( timeType != cond::timestamp ){
	return nextSince - 1;
      } else {
	UnpackedTime unpackedTime = unpack(  nextSince );
	//number of seconds in nanoseconds (avoid multiply and divide by 1e09)
	Time_t totalSecondsInNanoseconds = ((Time_t)unpackedTime.first)*1000000000;
	//total number of nanoseconds
	Time_t totalNanoseconds = totalSecondsInNanoseconds + ((Time_t)(unpackedTime.second));
	//now decrementing of 1 nanosecond
	totalNanoseconds--;
	//now repacking (just change the value of the previous pair)
	unpackedTime.first = (unsigned int) (totalNanoseconds/1000000000);
	unpackedTime.second = (unsigned int)(totalNanoseconds - (Time_t)unpackedTime.first*1000000000);
	return pack(unpackedTime);
      }
    }

  }

}
