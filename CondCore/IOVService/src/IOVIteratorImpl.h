#ifndef CondCore_IOVService_IOVIteratorImpl_h
#define CondCore_IOVService_IOVIteratorImpl_h
#include <string>
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
namespace cond{
  class PoolTransaction;
  class IOV;
  class IOVIteratorImpl : virtual public cond::IOVIterator{
  public:
    IOVIteratorImpl( cond::PoolTransaction& pooldb,
		     const std::string token , 
		     cond::Time_t globalSince, 
		     cond::Time_t globalTill);
    virtual ~IOVIteratorImpl();
    virtual bool next();
    virtual std::string payloadToken() const;
    virtual std::pair<cond::Time_t, cond::Time_t> validity() const;
  private:
    void init();
    cond::PoolTransaction& m_pooldb;
    std::string m_token;
    cond::Time_t m_globalSince;
    cond::Time_t m_globalTill;
    cond::TypedRef<cond::IOV> m_iov;
    size_t m_currentPos;
    size_t m_stop;
    bool m_isOpen;
  };
}//ns cond
#endif
