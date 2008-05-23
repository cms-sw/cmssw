#include "CondCore/IOVService/interface/IOVService.h"
#include "IOVServiceImpl.h"
#include "IOVIteratorImpl.h"
#include "IOVRevIteratorImpl.h"
#include "IOVEditorImpl.h"
cond::IOVService::IOVService( cond::PoolTransaction& pooldb):
  m_pooldb(&pooldb),
  m_impl(new cond::IOVServiceImpl(pooldb){
}
cond::IOVService::~IOVService(){
  delete m_impl;
}
std::string 
cond::IOVService::payloadToken( const std::string& iovToken,
				cond::Time_t currenttime ){
  return m_impl->payloadToken(iovToken, currenttime);
}
bool 
cond::IOVService::isValid( const std::string& iovToken,
			   cond::Time_t currenttime ){
  return m_impl->isValid(iovToken,currenttime);
}
std::pair<cond::Time_t, cond::Time_t> 
cond::IOVService::validity( const std::string& iovToken, cond::Time_t currenttime ){
  return m_impl->validity(iovToken,currenttime);
}
std::string 
cond::IOVService::payloadContainerName( const std::string& token ){
  return m_impl->payloadContainerName(token);
}
void 
cond::IOVService::deleteAll( bool withPayload ){
  m_impl->deleteAll( withPayload );
}
cond::IOVIterator* 
cond::IOVService::newIOVIterator( const std::string& token, bool forward ){
  return forward ?
    (cond::IOVIterator*)new cond::IOVIteratorImpl( *m_pooldb,token) :
    (cond::IOVIterator*)new cond::IOVRevIteratorImpl( *m_pooldb,token);
}

cond::IOVEditor* 
cond::IOVService::newIOVEditor( const std::string& token ){
  return new cond::IOVEditorImpl( *m_pooldb,token);
}


std::string 
cond::IOVService::exportIOVWithPayload( cond::PoolTransaction& destDB,
					const std::string& iovToken ){
  return m_impl->exportIOVWithPayload( destDB,
				       iovToken); 
}

std::string
cond::IOVService::exportIOVRangeWithPayload( cond::PoolTransaction& destDB,
					     const std::string& iovToken,
					     const std::string& destToken,
					     cond::Time_t since,
					     cond::Time_t till){
  return  m_impl->exportIOVRangeWithPayload( destDB,
					     iovToken,
					     destToken,
					     since,
					     till); 
}

