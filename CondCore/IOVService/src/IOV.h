#ifndef IOVService_IOV_h
#define IOVService_IOV_h
#include "CondCore/DBCommon/interface/Time.h"
#include <map>
#include <string>
namespace cond{
  class IOV {
  public:
    IOV(){}
    virtual ~IOV(){}
    //std::map<unsigned long long,std::string> iov;
    std::map<cond::Time_t,std::string> iov;
  };
}//ns cond
#endif
