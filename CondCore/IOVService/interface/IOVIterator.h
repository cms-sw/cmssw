#ifndef CondCore_IOVService_IOVIterator_h
#define CondCore_IOVService_IOVIterator_h
#include <string>
#include "CondCore/DBCommon/interface/Time.h"
namespace cond{
  class IOVIterator{
  public:
    virtual ~IOVIterator(){}
    virtual bool next()=0;
    virtual bool empty() const=0;
    virtual size_t size() const=0;
    virtual size_t remaining() const=0;
    virtual bool atEnd() const;
    virtual std::string payloadToken() const=0;   
    virtual cond::ValidityInterval validity() const=0;
  protected:
    IOVIterator(){}
  };
}//ns cond
#endif
