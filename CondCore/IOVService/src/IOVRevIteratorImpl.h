#ifndef CondCore_IOVService_IOVRevIteratorImpl_h
#define CondCore_IOVService_IOVRevIteratorImpl_h

#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Common/interface/IOVSequence.h"
#include <string>

namespace cond{
  class PoolTransaction;
  class IOVRevIteratorImpl : virtual public cond::IOVIterator{
  public:
    typedef IOVSequence::Container::const_reverse_iterator const_iterator;
    IOVRevIteratorImpl( cond::PoolTransaction& pooldb,
		     const std::string & token);
    virtual ~IOVRevIteratorImpl();
    virtual bool next();
    virtual bool rewind();
    virtual bool empty() const;
    virtual size_t size() const;
    virtual size_t position() const;
    virtual bool atEnd() const;
    virtual std::string payloadToken() const;
    TimeType timetype() const {
      open();
      return (TimeType)(m_iov->timetype);
  }

    virtual cond::ValidityInterval validity() const;
  private:
    void open() const;
    void init();
    IOVSequence::Container const & iov() const { return m_iov->iovs();}

    cond::PoolTransaction& m_pooldb;
    std::string m_token;
    cond::TypedRef<cond::IOVSequence> m_iov;
    const_iterator m_pos;
    const_iterator m_next;
    size_t m_count;
    cond::Time_t m_till;

    bool m_isInit;
    bool m_isOpen;

  };
}//ns cond
#endif
