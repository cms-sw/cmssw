#ifndef CondCore_IOVService_IOVIteratorImpl_h
#define CondCore_IOVService_IOVIteratorImpl_h
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "IOV.h"
#include <string>

namespace cond{
  class PoolTransaction;
  class IOVIteratorImpl : virtual public cond::IOVIterator{
  public:
    typedef IOV::Container::const_iterator const_iterator;
    IOVIteratorImpl( cond::PoolTransaction& pooldb,
		     const std::string & token , 
		     cond::Time_t globalSince, 
		     cond::Time_t globalTill);
    virtual ~IOVIteratorImpl();
    virtual bool next();
    virtual bool rewind();
    virtual bool empty() const;
    virtual size_t size() const;
    virtual size_t remaining() const;
    virtual bool atEnd() const;
    virtual std::string payloadToken() const;
    virtual cond::ValidityInterval validity() const;
  private:
    void init();
    cond::PoolTransaction& m_pooldb;
    std::string m_token;
    cond::Time_t m_globalSince;
    cond::Time_t m_globalTill;
    cond::TypedRef<cond::IOV> m_iov;
    const_iterator m_pos;
    cond::Time_t = m_since;
    size_t m_count;

    bool m_isOpen;
  };
}//ns cond
#endif
