#ifndef CondCore_IOVService_IOVRevIteratorImpl_h
#define CondCore_IOVService_IOVRevIteratorImpl_h

#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "IOV.h"
#include <string>

namespace cond{
  class PoolTransaction;
  class IOVRevIteratorImpl : virtual public cond::IOVIterator{
  public:
    typedef IOV::Container::const_reverse_iterator const_iterator;
    IOVRevIteratorImpl( cond::PoolTransaction& pooldb,
		     const std::string & token , 
		     cond::Time_t globalSince, 
		     cond::Time_t globalTill);
    virtual ~IOVRevIteratorImpl();
    virtual bool next();
    virtual bool goLast();
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
    const_iterator m_next;
    size_t m_count;

    bool m_isOpen;
  };
}//ns cond
#endif
