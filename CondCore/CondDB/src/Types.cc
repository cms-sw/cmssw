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

  static std::pair<const char *, SynchronizationType> s_synchronizationTypeArray[] = { std::make_pair("Offline", OFFLINE),
                                                                                       std::make_pair("HLT", HLT),
                                                                                       std::make_pair("Prompt", PROMPT),
                                                                                       std::make_pair("PCL", PCL) };
  std::string synchronizationTypeNames(SynchronizationType type) {
    return s_synchronizationTypeArray[type].first;
  }

  SynchronizationType synchronizationTypeFromName( const std::string& name ){
    for (auto const &i : s_synchronizationTypeArray)
      if (name.compare(i.first))
        return i.second;
    throwException( "SynchronizationType \""+name+"\" is unknown.","synchronizationTypeFromName");
  }

}
