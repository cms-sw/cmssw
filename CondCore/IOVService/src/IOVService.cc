#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/ContainerIterator.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "POOLCore/Token.h"

#include "CondFormats/Common/interface/PayloadWrapper.h"
#include "CondCore/DBCommon/interface/IOVInfo.h"




cond::IOVService::IOVService(cond::DbSession& pooldb):
  m_pooldb(pooldb) {}

cond::IOVService::~IOVService(){}

cond::IOVEditor* 
cond::IOVService::newIOVEditor( const std::string& token ){
  return new cond::IOVEditor( m_pooldb,token);
}




cond::IOVSequence const & cond::IOVService::iovSeq(const std::string& iovToken) {
  if (m_token!=iovToken) {
    pool::Ref<cond::IOVSequence> temp = m_pooldb.getTypedObject<cond::IOVSequence>( iovToken );
    m_iov.copyShallow(temp);
    m_token=iovToken;
  }
  return *m_iov;
}



std::string 
cond::IOVService::payloadToken( const std::string& iovToken,
				cond::Time_t currenttime ){
  cond::IOVSequence const & iov=iovSeq(iovToken);
  cond::IOVSequence::const_iterator iEnd=iov.find(currenttime);
  if( iEnd==iov.iovs().end() ){
    return "";
  }else{
    return iEnd->wrapperToken();
  }
}

bool cond::IOVService::isValid( const std::string& iovToken,
				cond::Time_t currenttime ){
  
  cond::IOVSequence const & iov=iovSeq(iovToken);
  return (  currenttime >= iov.firstSince() && 
	    currenttime <= iov.lastTill() );
}

std::pair<cond::Time_t, cond::Time_t> 
cond::IOVService::validity( const std::string& iovToken, cond::Time_t currenttime ){
  
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
cond::IOVService::payloadContainerName( const std::string& iovToken ){
  cond::IOVSequence const & iov=iovSeq(iovToken);
  
  // FIXME move to metadata
  std::string payloadtokstr=iov.iovs().front().wrapperToken();
  pool::Token theTok;
  theTok.fromString(payloadtokstr);
  return theTok.contID();
}




void 
cond::IOVService::deleteAll(bool withPayload){
  cond::ContainerIterator<cond::IOVSequence> it(m_pooldb,cond::IOVNames::container());
  while ( it.next() ) {
    if(withPayload){
      std::string tokenStr;
      IOVSequence::const_iterator payloadIt;
      IOVSequence::const_iterator payloadItEnd=it.dataRef()->iovs().end();
      for(payloadIt=it.dataRef()->iovs().begin();payloadIt!=payloadItEnd;++payloadIt){
        tokenStr=payloadIt->wrapperToken();
        m_pooldb.deleteObject( tokenStr );
      }
    }
    it.dataRef().markDelete();
  }
}

std::string
cond::IOVService::exportIOVWithPayload( cond::DbSession& destDB,
                                            const std::string& iovToken){
  cond::IOVSequence const & iov=iovSeq(iovToken);
  
  cond::IOVSequence* newiov=new cond::IOVSequence(iov.timeType(), iov.lastTill(),iov.metadataToken());
  
  for( IOVSequence::const_iterator it=iov.iovs().begin();
       it!=iov.iovs().end(); ++it){
    std::string newPToken = destDB.importObject( m_pooldb, it->wrapperToken());
    newiov->add(it->sinceTime(),newPToken);
  }
  return destDB.storeObject(newiov, cond::IOVNames::container()).toString();
}

#include "CondCore/DBCommon/interface/ClassInfoLoader.h"

void cond::IOVService::loadDicts( const std::string& iovToken) {
  // loadlib
  pool::Ref<cond::IOVSequence> iov = m_pooldb.getTypedObject<cond::IOVSequence>(iovToken);
  // FIXME use iov metadata
  std::string ptok = iov->iovs().front().wrapperToken();
  m_pooldb.transaction().commit();
  cond::reflexTypeByToken(ptok);
  m_pooldb.transaction().start(true);
}


std::string 
cond::IOVService::exportIOVRangeWithPayload( cond::DbSession& destDB,
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
  
  
  // since > ifirstTill->sinceTime() used to overwrite the actual time
  //since = ifirstTill->sinceTime();
  
  pool::Ref<cond::IOVSequence> newiovref;
  //FIXME more option and eventually ability to resume (roll back is difficult)
  std::string dToken = destToken;
  if (dToken.empty()) {
    // create a new one 
    newiovref = destDB.storeObject( new cond::IOVSequence(iov.timeType(), iov.lastTill(),iov.metadataToken()),cond::IOVNames::container());
    dToken = newiovref.toString();
  } else {
    newiovref = destDB.getTypedObject<cond::IOVSequence>(destToken);
    
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
  cond::Time_t lsince = since;
  for(  IOVSequence::const_iterator it=ifirstTill;
	it!=isecondTill; ++it, lsince=it->sinceTime()){
    // first since overwritten by global since...
    
    // FIXME need option to load Ptr unconditionally....
    pool::Ref<cond::PayloadWrapper> payloadTRef = m_pooldb.getTypedObject<cond::PayloadWrapper>(it->wrapperToken());
    if(payloadTRef.ptr()) payloadTRef->loadAll();
    std::string newPtoken = destDB.importObject( m_pooldb,it->wrapperToken());
    newiovref->add(lsince, newPtoken);
    /// commit to avoid HUGE memory footprint
    n++;
    if (n==100) {
      std::cout << "committing " << std::endl;
      n=0;
      destDB.transaction().commit();
      destDB.transaction().start(false);
      newiovref = destDB.getTypedObject<cond::IOVSequence>( dToken );
      newiovref.markUpdate();
    }
  }
  newiovref->stamp(cond::userInfo(),false);
  return newiovref.toString();
}

