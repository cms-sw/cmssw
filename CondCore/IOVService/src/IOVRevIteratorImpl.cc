#include "IOVRevIteratorImpl.h"
#include <map>
cond::IOVRevIteratorImpl::IOVRevIteratorImpl( cond::PoolTransaction& pooldb,
					const std::string & token,
					cond::Time_t globalSince, 
					cond::Time_t globalTill)
  : m_pooldb(pooldb),m_token(token), m_globalSince(globalSince),m_globalTill(globalTill),  m_count(0), m_isOpen(false){
} 
cond::IOVRevIteratorImpl::~IOVRevIteratorImpl(){
}
void cond::IOVRevIteratorImpl::init(){
  m_iov=cond::TypedRef<cond::IOV>(m_pooldb, m_token);
  m_isOpen=true;
  m_pos=m_iov->iov.rbegin();
  m_next=m_pos; m_next++;
  m_count = size();
}


bool cond::IOVRevIteratorImpl::rewind() {
  init();
  return !empty();
}

bool cond::IOVRevIteratorImpl::empty() const {
  return m_iov->iov.empty();
}
size_t cond::IOVRevIteratorImpl::size() const {
  return m_iov->iov.size();
}
size_t cond::IOVRevIteratorImpl::remaining() const {
  return size()-m_count;
}


bool  cond::IOVRevIteratorImpl::atEnd() const {
  return m_pos==m_iov->iov.rend();
}

bool cond::IOVRevIteratorImpl::next(){
  if(!m_isOpen){
    init();
    return !empty();
  }
  if (atEnd() ) return false;
  m_pos=m_next;
  if (atEnd() ) return false;
  ++m_next;
  --m_count;
  return true;
}

std::string 
cond::IOVRevIteratorImpl::payloadToken() const{
  if(!m_isOpen){
    const_cast<cond::IOVRevIteratorImpl*>(this)->init();
  }
  
  return atEnd() ? std::string("") : m_pos->second;

}

cond::ValidityInterval
cond::IOVRevIteratorImpl::validity() const{
  if(!m_isOpen){
    const_cast<cond::IOVRevIteratorImpl*>(this)->init();
  }
  cond::Time_t since=m_globalSince;
  cond::Time_t till=m_globalTill;
  if (!atEnd()) {
    till = m_pos->first;
    if (m_next!=m_iov->iov.rend()) since =  m_next->first;
  }
  return cond::ValidityInterval(since,till);
}
