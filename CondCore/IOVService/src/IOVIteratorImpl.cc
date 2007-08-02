#include "IOVIteratorImpl.h"
#include "IOV.h"
#include <map>
cond::IOVIteratorImpl::IOVIteratorImpl( cond::PoolTransaction& pooldb,
					const std::string token,
					cond::Time_t globalSince, 
					cond::Time_t globalTill)
  : m_pooldb(pooldb),m_token(token), m_globalSince(globalSince),m_globalTill(globalTill), m_currentPos(0), m_stop(0), m_isOpen(false){
} 
cond::IOVIteratorImpl::~IOVIteratorImpl(){
}
void cond::IOVIteratorImpl::init(){
  m_iov=cond::TypedRef<cond::IOV>(m_pooldb, m_token);
  m_stop=(m_iov->iov.size())-1;
  m_isOpen=true;
}
bool cond::IOVIteratorImpl::next(){
  if(!m_isOpen){
    init();
  }
  if(m_currentPos>m_stop){
    return false;
  }
  ++m_currentPos;
  return true;
}
std::string 
cond::IOVIteratorImpl::payloadToken() const{
  if(!m_isOpen){
    const_cast<cond::IOVIteratorImpl*>(this)->init();
  }
  size_t pos=1;
  for( std::map<cond::Time_t, std::string>::const_iterator it=m_iov->iov.begin(); it!=m_iov->iov.end(); ++it,++pos ){
    if(m_currentPos==pos){
      return it->second;
    }
  }
  return "";
}
std::pair<cond::Time_t, cond::Time_t> 
cond::IOVIteratorImpl::validity() const{
  if(!m_isOpen){
    const_cast<cond::IOVIteratorImpl*>(this)->init();
  }
  size_t pos=1;
  cond::Time_t since=m_globalSince;
  cond::Time_t till=m_globalTill;
  std::map<cond::Time_t, std::string>::iterator itbeg=m_iov->iov.begin();
  for(std::map<cond::Time_t, std::string>::iterator it=itbeg;
      it!=m_iov->iov.end();++it,++pos){
    if(pos==m_currentPos){
      till=it->first;
      if(m_currentPos != 1 ){
	--it;
	since=(it->first)+m_globalSince;
	++it;
      }
    }
  }
  return std::make_pair<cond::Time_t, cond::Time_t>(since,till);
}
