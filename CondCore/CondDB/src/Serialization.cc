#include "CondCore/CondDB/interface/Serialization.h"
//
#include <sstream>
#include "boost/version.hpp"

std::string cond::StreamerInfo::techVersion(){
  return BOOST_LIB_VERSION;
}

std::string cond::StreamerInfo::jsonString(){
  std::stringstream ss;
  ss<<" {"<<std::endl;
  ss<<"\""<<CMSSW_VERSION_LABEL<<"\": \""<<currentCMSSWVersion()<<"\","<<std::endl;
  ss<<"\""<<ARCH_LABEL<<"\": \""<<currentArchitecture()<<"\","<<std::endl;
  ss<<"\""<<TECH_LABEL<<"\": \""<<TECHNOLOGY<<"\","<<std::endl;
  ss<<"\""<<TECH_VERSION_LABEL<<"\": \""<<techVersion()<<"\""<<std::endl;
  ss<<" }"<<std::endl;
  return ss.str();
}

