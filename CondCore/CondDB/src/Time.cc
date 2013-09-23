#include "CondCore/CondDB/interface/Time.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Types.h"
//
#include <initializer_list>
#include <vector>
#include <map>

namespace conddb {

  namespace time {
    static auto s_timeTypeMap = { enumPair( "Run",RUNNUMBER ),
				  enumPair( "Time",TIMESTAMP ),
				  enumPair( "Lumi",LUMIID ),
				  enumPair( "Hash",HASH ),
				  enumPair( "User",USERID ) };

    std::string timeTypeName(TimeType type) {
      std::vector<std::pair<const std::string,TimeType> > tmp( s_timeTypeMap );
      if( type==INVALID ) return "";
      return tmp[type].first;
    }

    TimeType timeTypeFromName( const std::string& name ){
      std::map<std::string,TimeType> tmp( s_timeTypeMap );
      auto t = tmp.find( name );
      if( t == tmp.end() ) throwException( "TimeType \""+name+"\" is unknown.","timeTypeFromName");
      return t->second;
    }
  }

}
