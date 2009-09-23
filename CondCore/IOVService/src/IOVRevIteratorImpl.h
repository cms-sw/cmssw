#ifndef CondCore_IOVService_IOVRevIteratorImpl_h
#define CondCore_IOVService_IOVRevIteratorImpl_h

#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondFormats/Common/interface/IOVSequence.h"
#include "DataSvc/Ref.h"
#include <string>

namespace cond{
  class IOVRevIteratorImpl : virtual public cond::IOVIterator{
  public:
    typedef IOVSequence::Container::const_reverse_iterator const_iterator;
    IOVRevIteratorImpl( cond::DbSession& pooldb,
                        const std::string & token);
    virtual ~IOVRevIteratorImpl();
    virtual bool next();
    virtual bool rewind();
    virtual bool empty() const;
    virtual size_t size() const;
    virtual size_t position() const;
    virtual bool atEnd() const;
    virtual std::string payloadToken() const;
    TimeType timetype() const;

    virtual cond::ValidityInterval validity() const;
  private:
    void open() const;
    void init();
    IOVSequence::Container const & iov() const { return m_iov->iovs();}

    mutable cond::DbSession m_pooldb;
    std::string m_token;
    pool::Ref<cond::IOVSequence> m_iov;
    const_iterator m_pos;
    const_iterator m_next;
    size_t m_count;
    cond::Time_t m_till;

    bool m_isInit;
    bool m_isOpen;

  };
}//ns cond
#endif
