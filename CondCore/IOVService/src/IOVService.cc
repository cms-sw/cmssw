#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"

#include "FWCore/Framework/interface/IOVSyncValue.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include "CondCore/DBCommon/interface/IOVInfo.h"

namespace cond {

boost::shared_ptr<cond::IOVSequence> loadIOV( cond::DbSession& dbSess, 
                                              const std::string& iovToken ){
  boost::shared_ptr<cond::IOVSequence> iov = dbSess.getTypedObject<cond::IOVSequence>( iovToken );
  // loading the lazy-loading Queryable vector...
  iov->loadAll();
  //**** temporary for the schema transition
  if( dbSess.isOldSchema() ){
    PoolTokenParser parser(  dbSess.storage() ); 
    iov->swapTokens( parser );
  }
  //****
  return iov;
}

std::string insertIOV( cond::DbSession& dbSess, 
                       const boost::shared_ptr<IOVSequence>& data,
                       bool swapOIds=false ){
  dbSess.createDatabase();
  // ***** TEMPORARY FOR TRANSITION PHASE
  if( swapOIds && dbSess.isOldSchema() ){
    PoolTokenWriter writer( dbSess.storage() );
    data->swapOIds( writer );
  }
  // *****
  return dbSess.storeObject( data.get(), cond::IOVNames::container());
}

void updateIOV( cond::DbSession& dbSess, 
                const boost::shared_ptr<IOVSequence>& data,
                const std::string& token ){
  // ***** TEMPORARY FOR TRANSITION PHASE
  if( dbSess.isOldSchema() ){
    PoolTokenWriter writer( dbSess.storage() );
    data->swapOIds( writer );
  }
  // *****
  dbSess.updateObject( data.get(), token );
}

}

cond::IOVService::IOVService(cond::DbSession& dbSess):
  m_dbSess(dbSess) {}

cond::IOVService::~IOVService(){}

cond::IOVEditor* 
cond::IOVService::newIOVEditor( const std::string& token ){
  return new cond::IOVEditor( m_dbSess,token);
}

cond::IOVSequence const & cond::IOVService::iovSeq(const std::string& iovToken) {
  if (m_token!=iovToken) {
    m_iov = loadIOV( m_dbSess, iovToken );
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
    return iEnd->token();
  }
}

cond::TimeType cond::IOVService::timeType( const std::string& iovToken ) {
  cond::IOVSequence const & iov=iovSeq(iovToken);
  return iov.timeType();
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
  return std::pair<cond::Time_t, cond::Time_t>(since,till);
}

std::set<std::string>  
cond::IOVService::payloadClasses( const std::string& iovToken ){
  cond::IOVSequence const & iov=iovSeq(iovToken);
  return iov.payloadClasses();
}

void 
cond::IOVService::deleteAll(bool withPayload){
  ora::Database& db = m_dbSess.storage();
  ora::Container cont = db.containerHandle( cond::IOVNames::container() );
  ora::ContainerIterator it = cont.iterator();
  while ( it.next() ) {
    if(withPayload){
      std::string tokenStr;
      IOVSequence::const_iterator payloadIt;
      boost::shared_ptr<cond::IOVSequence> iov = it.get<cond::IOVSequence>();
      IOVSequence::const_iterator payloadItBegin=iov->iovs().begin();
      IOVSequence::const_iterator payloadItEnd=iov->iovs().end();
      for(payloadIt=payloadItBegin;payloadIt!=payloadItEnd;++payloadIt){
        tokenStr=payloadIt->token();
        m_dbSess.deleteObject( tokenStr );
      }
    }
    cont.erase( it.itemId() );
  }
  cont.flush();
}

std::string
cond::IOVService::exportIOVWithPayload( cond::DbSession& destDB,
					const std::string& iovToken){
  cond::IOVSequence const & iov=iovSeq(iovToken);
  
  boost::shared_ptr<cond::IOVSequence> newiov(new cond::IOVSequence(iov.timeType(), iov.lastTill(),iov.metadata()));
  
  for( IOVSequence::const_iterator it=iov.iovs().begin();
       it!=iov.iovs().end(); ++it){
    std::string newPToken = destDB.importObject( m_dbSess, it->token());
    ora::OId poid;
    poid.fromString( newPToken );
    ora::Container cont = destDB.storage().containerHandle( poid.containerId() );
    newiov->add(it->sinceTime(),newPToken, cont.className());
  }
  return insertIOV( destDB, newiov, true );
}

std::string 
cond::IOVService::exportIOVRangeWithPayload( cond::DbSession& destDB,
					     const std::string& iovToken,
					     const std::string& destToken,
					     cond::Time_t since,
					     cond::Time_t till,
					     bool outOfOrder ){
  cond::IOVSequence const & iov=iovSeq(iovToken);
  since = std::max(since, iov.firstSince());
  IOVSequence::const_iterator ifirstTill=iov.find(since);
  IOVSequence::const_iterator isecondTill=iov.find(till);
  if( isecondTill!=iov.iovs().end() ) isecondTill++;
  
  if (ifirstTill==isecondTill) 
    throw cond::Exception("IOVServiceImpl::exportIOVRangeWithPayload Error: empty input range");
  
  // since > ifirstTill->sinceTime() used to overwrite the actual time
  //since = ifirstTill->sinceTime();
  
  boost::shared_ptr<cond::IOVSequence> newiovref;
  //FIXME more option and eventually ability to resume (roll back is difficult)
  std::string dToken = destToken;
  if (dToken.empty()) {
    // create a new one
   newiovref.reset( new cond::IOVSequence(iov.timeType(), iov.lastTill(),iov.metadata()));
    dToken = insertIOV( destDB, newiovref );
  } else {
    newiovref = loadIOV( destDB, destToken);
    if (newiovref->iovs().empty()) ; // do not waist time
    else if (outOfOrder) {
      for( IOVSequence::const_iterator it=ifirstTill;
	   it!=isecondTill; ++it)
	if (newiovref->exist(it->sinceTime()))
	  throw cond::Exception("IOVServiceImpl::exportIOVRangeWithPayload Error: since time already exists");
    } else if (since <= newiovref->iovs().back().sinceTime())
      throw cond::Exception("IOVServiceImpl::exportIOVRangeWithPayload Error: since time out of range, below last since");
   updateIOV( destDB, newiovref, destToken );
  }

  cond::Time_t lsince = since;
  for(  IOVSequence::const_iterator it=ifirstTill;
	it!=isecondTill; ++it, lsince=it->sinceTime()){
    // first since overwritten by global since...
    
    std::string newPtoken = destDB.importObject( m_dbSess,it->token());
    ora::OId poid;
    poid.fromString( newPtoken );
    ora::Container cont = destDB.storage().containerHandle( poid.containerId() );
    newiovref->add(lsince, newPtoken,cont.className());
  }
  newiovref->stamp(cond::userInfo(),false);
  updateIOV( destDB, newiovref, dToken );
  return dToken;
}

