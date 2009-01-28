#include "IOVRevIteratorImpl.h"
#include<utility>
cond::IOVRevIteratorImpl::IOVRevIteratorImpl( cond::PoolTransaction& pooldb,
					const std::string & token)
  : m_pooldb(pooldb),m_token(token), m_count(0),  m_isInit(false), m_isOpen(false){
} 
cond::IOVRevIteratorImpl::~IOVRevIteratorImpl(){
}

void cond::IOVRevIteratorImpl::open() const{
  if (m_isOpen) return;
  const_cast<cond::IOVRevIteratorImpl*>(this)->m_iov=
    cond::TypedRef<cond::IOVSequence>(m_pooldb, m_token);
  const_cast<cond::IOVRevIteratorImpl*>(this)->m_isOpen=true;
}

void cond::IOVRevIteratorImpl::init(){
  open();
  m_isInit=true;
  m_pos=iov().rbegin();
  m_till = m_iov->lastTill();
  m_count = empty() ? 0 : size()-1;
}


bool cond::IOVRevIteratorImpl::rewind() {
  init();
  return !empty();
}

bool cond::IOVRevIteratorImpl::empty() const {
  open();
  return iov().empty();
}
size_t cond::IOVRevIteratorImpl::size() const {
  open();
  return iov().size();
}
size_t cond::IOVRevIteratorImpl::position() const {
  return m_count;
}


bool  cond::IOVRevIteratorImpl::atEnd() const {
  return  m_isInit && m_pos==iov().rend();
}

bool cond::IOVRevIteratorImpl::next(){
  if(!m_isInit){
    init();
    return !empty();
  }
  if (atEnd() ) return false;
  m_till=m_pos->sinceTime()-1;
  m_pos++;
  if (atEnd() ) return false;
  --m_count;
  return true;
}

std::string 
cond::IOVRevIteratorImpl::payloadToken() const{
  if(!m_isInit) return std::string("");
  
  return atEnd() ? std::string("") : m_pos->wrapperToken();
  
}

cond::ValidityInterval
cond::IOVRevIteratorImpl::validity() const{
  cond::Time_t since=0;
  cond::Time_t till=0;
  if (m_isInit && !atEnd()) {
    till = m_till;
    since = m_pos->sinceTime();
  }
  return cond::ValidityInterval(since,till);
}
