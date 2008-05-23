#include "IOVIteratorImpl.h"

cond::IOVIteratorImpl::IOVIteratorImpl( cond::PoolTransaction& pooldb,
					const std::string & token)
  : m_pooldb(pooldb),m_token(token), m_count(0), m_isInit(false), m_isOpen(false){
} 
cond::IOVIteratorImpl::~IOVIteratorImpl(){
}

void cond::IOVIteratorImpl::open() const{
  if (m_isOpen) return;
  const_cast<cond::IOVIteratorImpl*>(this)->m_iov=
    cond::TypedRef<cond::IOV>(m_pooldb, m_token);
  const_cast<cond::IOVIteratorImpl*>(this)->m_isOpen=true;
}
void cond::IOVIteratorImpl::init(){
  open();
  m_isInit=true;
  m_pos=m_iov->iov.begin();
  m_count = 0;
  m_since=m_iov->firstsince;
}


bool cond::IOVIteratorImpl::rewind() {
  init();
  return !empty();
}

bool cond::IOVIteratorImpl::empty() const {
  open();
  return m_iov->iov.empty();
}
size_t cond::IOVIteratorImpl::size() const {
  open();
  return m_iov->iov.size();
}
size_t cond::IOVIteratorImpl::position() const {
  return m_count;
}


bool  cond::IOVIteratorImpl::atEnd() const {
  return m_isInit && m_pos==m_iov->iov.end();
}

bool cond::IOVIteratorImpl::next(){
  if(!m_isInit){
    init();
    return !empty();
  }
  if (atEnd() ) return false;

  m_since = m_pos->first+1;
  ++m_pos;
  if (atEnd() ) return false;
  ++m_count;
  return true;
}

std::string 
cond::IOVIteratorImpl::payloadToken() const{
  if(!m_isInit) return std::string("");
  
  return atEnd() ? std::string("") : m_pos->second;

}

cond::ValidityInterval
cond::IOVIteratorImpl::validity() const{
  cond::Time_t since=0;
  cond::Time_t till=0;
  if (m_isInit && !atEnd()) {
    since = m_since;
    till =  m_pos->first;
  }
  return cond::ValidityInterval(since,till);
}
