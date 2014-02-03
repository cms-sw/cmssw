#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include <initializer_list>
#include <vector>
#include <map>

namespace cond {

  namespace time {
    static auto s_timeTypeMap = { enumPair( "Run",cond::runnumber ),
				  enumPair( "Time",cond::timestamp ),
				  enumPair( "Lumi",cond::lumiid ),
				  enumPair( "Hash",cond::hash ),
				  enumPair( "User",cond::userid ) };

    std::string timeTypeName(TimeType type) {
      std::vector<std::pair<const std::string,TimeType> > tmp( s_timeTypeMap );
      if( type==invalid ) return "";
      return tmp[type].first;
    }

    TimeType timeTypeFromName( const std::string& name ){
      std::map<std::string,TimeType> tmp( s_timeTypeMap );
      auto t = tmp.find( name );
      if( t == tmp.end() ) throwException( "TimeType \""+name+"\" is unknown.","timeTypeFromName");
      return t->second;
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
