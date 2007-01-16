#ifndef CondCore_IOVService_IOVIterator_h
#define CondCore_IOVService_IOVIterator_h
#include <string>
#include "CondCore/DBCommon/interface/Time.h"
namespace cond{
  class IOVIterator{
  public:
    virtual ~IOVIterator(){}
    virtual bool next()=0;
    virtual std::string payloadToken() const=0;   
    virtual std::pair<cond::Time_t, cond::Time_t> validity() const=0;
  protected:
    IOVIterator(){}
  };
}//ns cond
#endif
