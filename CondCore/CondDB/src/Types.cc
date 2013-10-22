#include "CondCore/CondDB/interface/Types.h"
#include "CondCore/CondDB/interface/Exception.h"
//
#include <initializer_list>
#include <vector>
#include <map>

namespace cond {

  void Iov_t::clear(){
    since = time::MAX;
    till = time::MIN;
    payloadId.clear();
  }

  bool Iov_t::isValid() const {
    return since != time::MAX && till != time::MIN && !payloadId.empty();
  }

  bool Iov_t::isValidFor( Time_t target ) const {
    return target >= since && target <= till;
  }

  void Tag_t::clear(){
    tag.clear();
    payloadType.clear();
    timeType = invalid;
    endOfValidity = time::MIN;
    lastValidatedTime = time::MIN;
  }

  static auto s_synchronizationTypeMap = { enumPair( "Offline",OFFLINE ),
					   enumPair( "HLT",HLT ),
					   enumPair( "Prompt",PROMPT ),
					   enumPair( "PCL",PCL ) };

  std::string synchronizationTypeNames(SynchronizationType type) {
    std::vector<std::pair<const std::string, SynchronizationType> > tmp( s_synchronizationTypeMap );
    return tmp[type].first;
  }

  SynchronizationType synchronizationTypeFromName( const std::string& name ){
    std::map<std::string,SynchronizationType> tmp( s_synchronizationTypeMap );
    auto t = tmp.find( name );
    if( t == tmp.end() ) throwException( "SynchronizationType \""+name+"\" is unknown.","synchronizationTypeFromName");
    return t->second;
  }

}
