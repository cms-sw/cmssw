#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/ContainerIterator.h"
#include "CondCore/DBCommon/interface/GenericRef.h"
#include "IOVServiceImpl.h"
#include "IOVIteratorImpl.h"
#include "IOVEditorImpl.h"
#include "POOLCore/Token.h"

cond::IOVServiceImpl::IOVServiceImpl( cond::PoolTransaction& pooldb ,
				      cond::TimeType timetype ): 
  m_pooldb(&pooldb), m_timetype(timetype){
  switch (m_timetype) {
  case cond::runnumber:
    m_beginOftime=(cond::Time_t)edm::IOVSyncValue::beginOfTime().eventID().run();
    m_endOftime=(cond::Time_t)edm::IOVSyncValue::endOfTime().eventID().run();
    break;
  case cond::timestamp:
    m_beginOftime=(cond::Time_t)edm::IOVSyncValue::beginOfTime().time().value();
    m_endOftime=(cond::Time_t)edm::IOVSyncValue::endOfTime().time().value();
    break;
  default:
    m_beginOftime=(cond::Time_t)edm::IOVSyncValue::beginOfTime().eventID().run();
    m_endOftime=(cond::Time_t)edm::IOVSyncValue::endOfTime().eventID().run();
    break;
  }
}
cond::IOVServiceImpl::~IOVServiceImpl(){
}

std::string 
cond::IOVServiceImpl::payloadToken( const std::string& iovToken,
				    cond::Time_t currenttime ){
  std::map< std::string,cond::TypedRef<cond::IOV> >::iterator it=m_iovcache.find(iovToken);
  if(it==m_iovcache.end()){
    m_iovcache.insert(std::make_pair< std::string,cond::TypedRef<cond::IOV> >(iovToken,cond::TypedRef<cond::IOV>(*m_pooldb,iovToken)));
  }
  cond::TypedRef<cond::IOV> iov=m_iovcache[iovToken];
  cond::IOV::const_iterator iEnd=iov->find(currenttime);
  if( iEnd==iov->iov.end() ){
    return "";
  }else{
    return iEnd->second;
  }
}

bool cond::IOVServiceImpl::isValid( const std::string& iovToken,
				    cond::Time_t currenttime ){
  std::map< std::string,cond::TypedRef<cond::IOV> >::iterator it=m_iovcache.find(iovToken);
  if(it==m_iovcache.end()){
    m_iovcache.insert(std::make_pair< std::string,cond::TypedRef<cond::IOV> >(iovToken,cond::TypedRef<cond::IOV>(*m_pooldb,iovToken)));
  }
  cond::TypedRef<cond::IOV> iov=m_iovcache[iovToken];
  bool result;
  if(  currenttime >= iov->firstsince && 
       currenttime <= iov->iov.back().first ){
    result=true;
  }else{
    result=false;
  }
  return result;
}

std::pair<cond::Time_t, cond::Time_t> 
cond::IOVServiceImpl::validity( const std::string& iovToken, cond::Time_t currenttime ){
  std::map< std::string,cond::TypedRef<cond::IOV> >::iterator it=m_iovcache.find(iovToken);
  if(it==m_iovcache.end()){
    m_iovcache.insert(std::make_pair< std::string,cond::TypedRef<cond::IOV> >(iovToken,cond::TypedRef<cond::IOV>(*m_pooldb,iovToken)));
  }
  cond::TypedRef<cond::IOV> iov=m_iovcache[iovToken];

  cond::Time_t since=iov->firstsince;
  cond::Time_t till=m_endOftime;
  IOV::const_iterator iter=iov->find(currenttime);
  if (iter!=iov->iov.end()) till=iter->first;
  if( iter!=iov->iov.begin() ){
    --iter; 
    since=iter->first+m_beginOftime;
  }
  return std::make_pair<cond::Time_t, cond::Time_t>(since,till);
}

std::string 
cond::IOVServiceImpl::payloadContainerName( const std::string& iovToken ){
  std::map< std::string,cond::TypedRef<cond::IOV> >::iterator it=m_iovcache.find(iovToken);
  if(it==m_iovcache.end()){
    m_iovcache.insert(std::make_pair< std::string,cond::TypedRef<cond::IOV> >(iovToken,cond::TypedRef<cond::IOV>(*m_pooldb,iovToken)));
  }
  cond::TypedRef<cond::IOV> iov=m_iovcache[iovToken];
  std::string payloadtokstr=iov->iov.begin()->second;
  pool::Token* theTok = new pool::Token;
  theTok->fromString(payloadtokstr);
  std::string result=theTok->contID();
  theTok->release();
  return result;
}

void 
cond::IOVServiceImpl::deleteAll(bool withPayload){
  cond::ContainerIterator<cond::IOV> it(*m_pooldb,cond::IOVNames::container());
  while ( it.next() ) {
    if(withPayload){
      std::string tokenStr;
      IOV::iterator payloadIt;
      IOV::iterator payloadItEnd=it.dataRef()->iov.end();
      for(payloadIt=it.dataRef()->iov.begin();payloadIt!=payloadItEnd;++payloadIt){
	tokenStr=payloadIt->second;
	pool::Token token;
	const pool::Guid& classID=token.fromString(tokenStr).classID();
	cond::GenericRef ref(*m_pooldb,tokenStr,pool::DbReflex::forGuid(classID).TypeInfo());
	ref.markDelete();
	ref.reset();
      }
    }
    it.dataRef().markDelete();
  }
}

cond::TimeType 
cond::IOVServiceImpl::timeType() const{
  return m_timetype;
}

cond::Time_t 
cond::IOVServiceImpl::globalSince() const{
  return m_beginOftime;
}

cond::Time_t 
cond::IOVServiceImpl::globalTill() const{
  return m_endOftime;
}

std::string
cond::IOVServiceImpl::exportIOVWithPayload( cond::PoolTransaction& destDB,
					    const std::string& iovToken){

  std::map< std::string,cond::TypedRef<cond::IOV> >::iterator it=m_iovcache.find(iovToken);
  if(it==m_iovcache.end()){
    m_iovcache.insert(std::make_pair< std::string,cond::TypedRef<cond::IOV> >(iovToken,cond::TypedRef<cond::IOV>(*m_pooldb,iovToken)));
  }

  cond::TypedRef<cond::IOV> iov=m_iovcache[iovToken];
  cond::IOV* newiov=new cond::IOV;
  newiov->timetype= iov->timetype;
  newiov->firstsince=iov->firstsince;

  for( IOV::iterator it=iov->iov.begin();
       it!=iov->iov.end(); ++it){
    cond::GenericRef payloadRef(*m_pooldb,it->second);
    std::string newPToken=payloadRef.exportTo(destDB);
    newiov->add(it->first,newPToken);
  }
  cond::TypedRef<cond::IOV> newiovref(destDB,newiov);
  newiovref.markWrite(cond::IOVNames::container());
  return newiovref.token();
}

std::string 
cond::IOVServiceImpl::exportIOVRangeWithPayload( cond::PoolTransaction& destDB,
						 const std::string& iovToken,
						 const std::string& destToken,
						 cond::Time_t since,
						 cond::Time_t till){

  std::map< std::string,cond::TypedRef<cond::IOV> >::iterator it=m_iovcache.find(iovToken);

  if(it==m_iovcache.end()){
    m_iovcache.insert(std::make_pair< std::string,cond::TypedRef<cond::IOV> >(iovToken,cond::TypedRef<cond::IOV>(*m_pooldb,iovToken)));
  }

  cond::TypedRef<cond::IOV> iov=m_iovcache[iovToken];
  IOV::const_iterator ifirstTill=iov->find(since);
  IOV::const_iterator isecondTill=iov->find(till);
  if( isecondTill!=iov->iov.end() ){
    isecondTill++;
  }
  
  if (ifirstTill==isecondTill) 
    throw cond::Exception("IOVServiceImpl::exportIOVRangeWithPayload Error: empty input range");


  IOV::const_iterator iprev=ifirstTill;

  // compute since
  since = std::max(since,(iprev==iov->iov.begin()) ? iov->firstsince : (--iprev)->first+1);

  cond::TypedRef<cond::IOV> newiovref;

  cond::Time_t lastIOV = m_endOftime;


  if (destToken.empty()) {
    // create a new one 
    newiovref = cond::TypedRef<cond::IOV>(destDB,new cond::IOV);
    newiovref.markWrite(cond::IOVNames::container());
    newiovref->timetype= iov->timetype;
    newiovref->firstsince = since;
  } else {
    newiovref = cond::TypedRef<cond::IOV>(destDB,destToken);
    newiovref.markUpdate();
    if (since <= newiovref->firstsince
	|| (newiovref->iov.size()>1 && since <= (++(newiovref->iov.rbegin()))->first)
	)  {
      throw cond::Exception("IOVServiceImpl::exportIOVRangeWithPayload Error: since time out of range, below last since");

    }
    // update last till
    lastIOV = newiovref->iov.back().first;
    newiovref->iov.back().first=since-1;
  }

  cond::IOV & newiov = *newiovref;
  for( IOV::const_iterator it=ifirstTill;
       it!=isecondTill; ++it){
    cond::GenericRef payloadRef(*m_pooldb,it->second);
    std::string newPtoken=payloadRef.exportTo(destDB);
    newiov.add(it->first,newPtoken);
  }
  // close (well open) IOV
  newiovref->iov.back().first = std::max(lastIOV, newiovref->iov.back().first);
  return newiovref.token();
}
