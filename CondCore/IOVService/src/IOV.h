#ifndef IOVService_IOV_h
#define IOVService_IOV_h
#include "CondCore/DBCommon/interface/Time.h"
#include <map>
#include <string>
namespace cond{
  class IOV {
  public:
    typedef IOV::Container Container;
    typedef Container::iterator iterator;
    typedef Container::const_iterator const_iterator;
    IOV(){}
    virtual ~IOV(){}
    //std::map<unsigned long long,std::string> iov;
    IOV::Container iov;
    int timetype;
    cond::Time_t firstsince;
  };
}//ns cond
#endif
