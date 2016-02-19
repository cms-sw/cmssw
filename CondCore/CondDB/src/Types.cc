#include "CondCore/CondDB/interface/Types.h"
#include "CondCore/CondDB/interface/Exception.h"
//
#include <initializer_list>
#include <vector>
#include <map>

namespace cond {

  template<size_t SIZE, class T> inline size_t array_size(T (&arr)[SIZE]){
    return SIZE;
  }

  void Iov_t::clear(){
    since = time::MAX_VAL;
    till = time::MIN_VAL;
    payloadId.clear();
  }

  bool Iov_t::isValid() const {
    return since != time::MAX_VAL && till != time::MIN_VAL && !payloadId.empty();
  }

  bool Iov_t::isValidFor( Time_t target ) const {
    return target >= since && target <= till;
  }

  void Tag_t::clear(){
    tag.clear();
    payloadType.clear();
    timeType = invalid;
    endOfValidity = time::MIN_VAL;
    lastValidatedTime = time::MIN_VAL;
  }

  static std::pair<const char *, SynchronizationType> s_synchronizationTypeArray[] = { std::make_pair("any", SYNCH_ANY),
										       std::make_pair("validation", SYNCH_VALIDATION),
										       std::make_pair("offline", SYNCH_OFFLINE),
										       std::make_pair("mc", SYNCH_MC),
										       std::make_pair("runmc", SYNCH_RUNMC),
										       std::make_pair("hlt", SYNCH_HLT),
										       std::make_pair("express", SYNCH_EXPRESS),
										       std::make_pair("prompt", SYNCH_PROMPT),
										       std::make_pair("pcl", SYNCH_PCL) };

  static std::pair<const char *, SynchronizationType> s_obsoleteSynchronizationTypeArray[] = { std::make_pair("Offline", SYNCH_OFFLINE),
											       std::make_pair("HLT", SYNCH_HLT),
											       std::make_pair("Prompt", SYNCH_PROMPT),
											       std::make_pair("Pcl", SYNCH_PCL) };
  std::string synchronizationTypeNames(SynchronizationType type) {
    if ( type>=array_size( s_synchronizationTypeArray ) )
      throwException( "SynchronizationType \""+boost::lexical_cast<std::string>(type)+"\" is unknown.","synchronizationTypeNames");      
    return s_synchronizationTypeArray[type].first;
  }

  SynchronizationType synchronizationTypeFromName( const std::string& name ){
    for (auto const &i : s_synchronizationTypeArray)
      if (name.compare(i.first)==0) return i.second;
    for (auto const &i : s_obsoleteSynchronizationTypeArray)
      if (name.compare(i.first)==0) return i.second;
    throwException( "SynchronizationType \""+name+"\" is unknown.","synchronizationTypeFromName");
  }

}

