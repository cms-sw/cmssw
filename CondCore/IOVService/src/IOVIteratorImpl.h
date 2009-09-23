#ifndef CondCore_IOVService_IOVIteratorImpl_h
#define CondCore_IOVService_IOVIteratorImpl_h
#include "CondCore/IOVService/interface/IOVIterator.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondFormats/Common/interface/IOVSequence.h"
#include "DataSvc/Ref.h"

#include <string>

namespace cond{
  class IOVIteratorImpl : virtual public cond::IOVIterator{
  public:
    typedef IOVSequence::const_iterator const_iterator;
    IOVIteratorImpl( cond::DbSession& pooldb,
		     const std::string & token);
    virtual ~IOVIteratorImpl();
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
    cond::Time_t tillTime() const;
    mutable cond::DbSession m_pooldb;
    std::string m_token;
    pool::Ref<cond::IOVSequence> m_iov;
    const_iterator m_pos;
    size_t m_count;

    bool m_isInit;
    bool m_isOpen;
  };
}//ns cond
#endif
