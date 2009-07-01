#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/ContainerIterator.h"
#include "CondCore/DBCommon/interface/GenericRef.h"
#include "IOVServiceImpl.h"
#include "IOVIteratorImpl.h"
#include "IOVEditorImpl.h"
#include "POOLCore/Token.h"

#include "CondFormats/Common/interface/PayloadWrapper.h"
#include "CondCore/DBCommon/interface/IOVInfo.h"


cond::IOVSequence const & cond::IOVServiceImpl::iovSeq(const std::string& iovToken) const {
  std::map< std::string,cond::IOVSequence const * >::const_iterator it=m_iovcache.find(iovToken);
  cond::IOVSequence const * iov=0;
  if(it==m_iovcache.end()){
    iov = new cond::IOVSequence(*cond::TypedRef<cond::IOVSequence>(*m_pooldb,iovToken));
    const_cast<cond::IOVServiceImpl*>(this)->m_iovcache.insert(std::make_pair(iovToken,iov));
  } else {
    iov = it->second;
  }
  return *iov;
}


cond::IOVServiceImpl::IOVServiceImpl( cond::PoolTransaction& pooldb) :
  m_pooldb(&pooldb) {
}

cond::IOVServiceImpl::~IOVServiceImpl(){
  for(std::map< std::string, cond::IOVSequence const* >::const_iterator iter= m_iovcache.begin();
      iter!=m_iovcache.end();++iter) delete const_cast<cond::IOVSequence*>(iter->second);
  
}

std::string 
cond::IOVServiceImpl::payloadToken( const std::string& iovToken,
				    cond::Time_t currenttime ){
  cond::IOVSequence const & iov=iovSeq(iovToken);
  cond::IOVSequence::const_iterator iEnd=iov.find(currenttime);
  if( iEnd==iov.iovs().end() ){
    return "";
  }else{
    return iEnd->wrapperToken();
  }
}

bool cond::IOVServiceImpl::isValid( const std::string& iovToken,
				    cond::Time_t currenttime ){

  cond::IOVSequence const & iov=iovSeq(iovToken);
  return (  currenttime >= iov.firstSince() && 
	    currenttime <= iov.lastTill() );
}

std::pair<cond::Time_t, cond::Time_t> 
cond::IOVServiceImpl::validity( const std::string& iovToken, cond::Time_t currenttime ){

  cond::IOVSequence const & iov=iovSeq(iovToken);

  cond::Time_t since=iov.firstSince();
  cond::Time_t till=iov.lastTill();
  IOVSequence::const_iterator iter=iov.find(currenttime);
  if (iter!=iov.iovs().end())  {
    since=iter->sinceTime();
    iter++;
    if (iter!=iov.iovs().end()) 
      till = iter->sinceTime()-1;
  }
  else {
    since=iov.lastTill();
  }
  return std::make_pair<cond::Time_t, cond::Time_t>(since,till);
}

std::string 
cond::IOVServiceImpl::payloadContainerName( const std::string& iovToken ){
  cond::IOVSequence const & iov=iovSeq(iovToken);

  // FIXME move to metadata
  std::string payloadtokstr=iov.iovs().front().wrapperToken();
  pool::Token theTok;
  theTok.fromString(payloadtokstr);
  return theTok.contID();
}

void 
cond::IOVServiceImpl::deleteAll(bool withPayload){
  cond::ContainerIterator<cond::IOVSequence> it(*m_pooldb,cond::IOVNames::container());
  while ( it.next() ) {
    if(withPayload){
      std::string tokenStr;
      IOVSequence::const_iterator payloadIt;
      IOVSequence::const_iterator payloadItEnd=it.dataRef()->iovs().end();
      for(payloadIt=it.dataRef()->iovs().begin();payloadIt!=payloadItEnd;++payloadIt){
	tokenStr=payloadIt->wrapperToken();
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

std::string
cond::IOVServiceImpl::exportIOVWithPayload( cond::PoolTransaction& destDB,
					    const std::string& iovToken){


  cond::IOVSequence const & iov=iovSeq(iovToken);

  cond::IOVSequence* newiov=new cond::IOVSequence(iov.timeType(), iov.lastTill(),iov.metadataToken());

  for( IOVSequence::const_iterator it=iov.iovs().begin();
       it!=iov.iovs().end(); ++it){
    cond::GenericRef payloadRef(*m_pooldb,it->wrapperToken());
    std::string newPToken=payloadRef.exportTo(destDB);
    newiov->add(it->sinceTime(),newPToken);
  }
  cond::TypedRef<cond::IOVSequence> newiovref(destDB,newiov);
  newiovref.markWrite(cond::IOVNames::container());
  return newiovref.token();
}


#include "CondCore/DBCommon/interface/ClassInfoLoader.h"

void cond::IOVServiceImpl::loadDicts( const std::string& iovToken) {
    // loadlib
    cond::TypedRef<cond::IOVSequence> iov(*m_pooldb,iovToken);
    // FIXME use iov metadata
    std::string ptok = iov->iovs().front().wrapperToken();
    m_pooldb->commit();   
    cond::reflexTypeByToken(ptok);
    m_pooldb->start(true);
}


std::string 
cond::IOVServiceImpl::exportIOVRangeWithPayload( cond::PoolTransaction& destDB,
						 const std::string& iovToken,
						 const std::string& destToken,
						 cond::Time_t since,
						 cond::Time_t till, 
						 bool outOfOrder){
 
  
  loadDicts(iovToken);

  cond::IOVSequence const & iov=iovSeq(iovToken);
  since = std::max(since, iov.firstSince());
  IOVSequence::const_iterator ifirstTill=iov.find(since);
  IOVSequence::const_iterator isecondTill=iov.find(till);
  if( isecondTill!=iov.iovs().end() ) isecondTill++;
  
  if (ifirstTill==isecondTill) 
    throw cond::Exception("IOVServiceImpl::exportIOVRangeWithPayload Error: empty input range");
  
  
  since = ifirstTill->sinceTime();
  
  cond::TypedRef<cond::IOVSequence> newiovref;
  //FIXME more option and eventually ability to resume (roll back is difficult)
  std::string dToken = destToken;
  if (dToken.empty()) {
    // create a new one 
    newiovref = 
      cond::TypedRef<cond::IOVSequence>(destDB,
					new cond::IOVSequence(iov.timeType(), iov.lastTill(),iov.metadataToken()));
    newiovref.markWrite(cond::IOVNames::container());
    dToken = newiovref.token();
  } else {
    newiovref = cond::TypedRef<cond::IOVSequence>(destDB,destToken);

    if (newiovref->iovs().empty()) ; // do not waist time
    else if (outOfOrder) {
        for( IOVSequence::const_iterator it=ifirstTill;
	     it!=isecondTill; ++it)
	  if (newiovref->exist(it->sinceTime()))
	    throw cond::Exception("IOVServiceImpl::exportIOVRangeWithPayload Error: since time already exists");
    } else if (since <= newiovref->iovs().back().sinceTime())
      throw cond::Exception("IOVServiceImpl::exportIOVRangeWithPayload Error: since time out of range, below last since");
    newiovref.markUpdate();    
  }


  int n=0;
  for( IOVSequence::const_iterator it=ifirstTill;
       it!=isecondTill; ++it){
    // FIXME need option to load Ptr unconditionally....
    cond::TypedRef<cond::PayloadWrapper> payloadTRef(*m_pooldb,it->wrapperToken());
    if(payloadTRef.ptr()) payloadTRef->loadAll();
    cond::GenericRef payloadRef(*m_pooldb,it->wrapperToken());
    std::string newPtoken=payloadRef.exportTo(destDB);
    newiovref->add(it->sinceTime(), newPtoken);
    /// commit to avoid HUGE memory footprint
    n++;
    if (n==100) {
      std::cout << "committing " << std::endl;
      n=0;
      destDB.commit();
      destDB.start(false);
      newiovref = cond::TypedRef<cond::IOVSequence>(destDB,dToken);
      newiovref.markUpdate();
    }
  }
  newiovref->stamp(cond::userInfo(),false);
  return newiovref.token();
}
