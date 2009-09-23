#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "IOVEditorImpl.h"
#include "CondFormats/Common/interface/IOVSequence.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbReflex.h"
#include<ostream>
#include<sstream>

namespace cond {

  IOVEditorImpl::IOVEditorImpl( cond::DbSession& poolDb,
				const std::string& token
				):m_poolDb(poolDb),m_token(token),
				  m_isActive(false){
  }
  
  IOVSequence & IOVEditorImpl::iov(){ return *m_iov;}


  void IOVEditorImpl::debugInfo(std::ostream & co) const {
    co << "IOVEditor: ";
    co << "db " << m_poolDb.connectionString();
    if(m_token.empty()) {
      co << " no token"; return;
    }
    if (!m_iov.ptr() )  {
      co << " no iov for token " << m_token;
      return;
    }
    co << " iov token " << m_token;
    co << "\nStamp: " <<  m_iov->comment()
       << "; time " <<  m_iov->timestamp()
       << "; revision " <<  m_iov->revision();
    co <<". TimeType " << cond::timeTypeSpecs[ m_iov->timeType()].name;
    if(  m_iov->iovs().empty() ) 
      co << ". empty";
    else
      co << ". size " <<  m_iov->iovs().size() 
	 << "; last since " << m_iov->iovs().back().sinceTime();
  }

  void IOVEditorImpl::reportError(std::string message) const {
    std::ostringstream out;
    out << "Error in ";
    debugInfo(out);
    out  << "\n" << message;
    throw cond::Exception(out.str());
  }

  void IOVEditorImpl::reportError(std::string message, cond::Time_t time) const {
    std::ostringstream out;
    out << "Error in";
    debugInfo(out);
    out << "\n" <<  message << " for time:  " << time;
    throw cond::Exception(out.str());
  }


  // create empty default sequence
  void IOVEditorImpl::create(cond::TimeType timetype) {
    if(!m_token.empty()){
      // problem??
      reportError("cond::IOVEditorImpl::create cannot create a IOV using an initialized Editor");
    }

    m_iov = m_poolDb.storeObject(new cond::IOVSequence(timetype),cond::IOVNames::container());
    m_token=m_iov.toString();
    m_isActive=true;

  }

  void IOVEditorImpl::create(cond::TimeType timetype,cond::Time_t lastTill) {

    if(!m_token.empty()){
      // problem??
      reportError("cond::IOVEditorImpl::create cannot create a IOV using an initialized Editor");
    }

    if(!validTime(lastTill, timetype))
      reportError("cond::IOVEditorImpl::create time not in global range",lastTill);

    m_iov = m_poolDb.storeObject(new cond::IOVSequence((int)timetype,lastTill," "),cond::IOVNames::container());
    m_token=m_iov.toString();
    m_isActive=true;
    
  }
  


  
  void IOVEditorImpl::init(){
    if(m_token.empty()){
      // problem?
      reportError("cond::IOVEditorImpl::init cannot init w/o token change");
    }
    
    m_iov = m_poolDb.getTypedObject<cond::IOVSequence>(m_token);
    m_isActive=true;
    
  }
  
  
  
  IOVEditorImpl::~IOVEditorImpl(){
  }
  
  Time_t IOVEditorImpl::firstSince() const {
    return m_iov->firstSince();
  }
  
  Time_t IOVEditorImpl::lastTill() const {
    return m_iov->lastTill();
  }
  
  TimeType IOVEditorImpl::timetype() const {
    return m_iov->timeType();
  }
  
  
  bool IOVEditorImpl::validTime(cond::Time_t time, cond::TimeType timetype) const {
    return time>=timeTypeSpecs[timetype].beginValue && time<=timeTypeSpecs[timetype].endValue;   
    
  }
  
  bool IOVEditorImpl::validTime(cond::Time_t time) const {
    return validTime(time,timetype());
  }
  
  
  
  unsigned int
  IOVEditorImpl::insert( cond::Time_t tillTime,
			 const std::string& payloadToken
			 ){
    if(!m_isActive) this->init();
    
    if( m_iov->iovs().empty() ) 
      reportError("cond::IOVEditorImpl::insert cannot inser into empty IOV sequence",tillTime);
    
    if(!validTime(tillTime))
      reportError("cond::IOVEditorImpl::insert time not in global range",tillTime);
    
    if(tillTime<=lastTill() )
      reportError("cond::IOVEditorImpl::insert IOV not in range",tillTime);
    
    cond::Time_t newSince=lastTill()+1;
    m_iov.markUpdate();
    updateClosure(tillTime);
    return m_iov->add(newSince, payloadToken);
    
  }
  
  void 
  IOVEditorImpl::bulkAppend(std::vector< std::pair<cond::Time_t,std::string> >& values){
    if (values.empty()) return;
    if(!m_isActive) this->init();
    cond::Time_t firstTime = values.front().first;
    cond::Time_t  lastTime = values.back().first;

    if(!validTime(firstTime))
      reportError("cond::IOVEditorImpl::bulkInsert first time not in global range",firstTime);

    if(!validTime(lastTime))
      reportError("cond::IOVEditorImpl::bulkInsert last time not in global range",lastTime);

   if(lastTime>=lastTill() ||
      ( !m_iov->iovs().empty() && firstTime<=m_iov->iovs().back().sinceTime()) 
       )    
     reportError("cond::IOVEditorImpl::bulkInsert IOV not in range",firstTime);

   for(std::vector< std::pair<cond::Time_t,std::string> >::const_iterator it=values.begin(); it!=values.end(); ++it){
     //     m_iov->iov.insert(m_iov->iov.end(), values.begin(), values.end());
     m_iov->add(it->first,it->second);
   }
    m_iov.markUpdate();   
  }

  void 
  IOVEditorImpl::bulkAppend(std::vector< cond::IOVElement >& values){
    if (values.empty()) return;
    if(!m_isActive) this->init();
    cond::Time_t firstTime = values.front().sinceTime();
    cond::Time_t   lastTime = values.back().sinceTime();

    if(!validTime(firstTime))
      reportError("cond::IOVEditorImpl::bulkInsert first time not in global range",firstTime);

    if(!validTime(lastTime))
      reportError("cond::IOVEditorImpl::bulkInsert last time not in global range",lastTime);

   if(lastTime>=lastTill() ||
      ( !m_iov->iovs().empty() && firstTime<=m_iov->iovs().back().sinceTime()) 
      )    reportError("cond::IOVEditorImpl::bulkInsert IOV not in range",firstTime);

   // m_iov->iov.insert(m_iov->iov.end(), values.begin(), values.end());
   m_iov.markUpdate();   
 
 }
  

  void 
  IOVEditorImpl::stamp(std::string const & icomment, bool append) {
    if(!m_isActive) this->init();
    m_iov->stamp(icomment, append);
    m_iov.markUpdate();
  }


  void 
  IOVEditorImpl::updateClosure( cond::Time_t newtillTime ){
    if( m_token.empty() ) reportError("cond::IOVEditorImpl::updateClosure cannot change non-existing IOV index");
    if(!m_isActive) this->init();
    m_iov->updateLastTill(newtillTime);
    m_iov.markUpdate();
  }
  
  unsigned int 
  IOVEditorImpl::append( cond::Time_t sinceTime,
			 const std::string&payloadToken
			 ){
    if( m_token.empty() ) {
      reportError("cond::IOVEditorImpl::appendIOV cannot append to non-existing IOV index");
    }
    
    if(!m_isActive) {
      this->init();
    }

    if(!validTime(sinceTime))
      reportError("cond::IOVEditorImpl::append time not in global range",sinceTime);
    
    
    if(  m_iov->iovs().size()>1 ){
      //range check in case 
      cond::Time_t lastValidSince=m_iov->iovs().back().sinceTime();
      //std::cout<<"lastValidTill "<<lastValidTill<<std::endl;
      if( sinceTime<= lastValidSince){
	reportError("IOVEditor::append Error: since time out of range: below last since",sinceTime);
      }
    }

    // does it make sense? (in case of mixed till and since insertions...)
    if (lastTill()<=sinceTime) updateClosure(timeTypeSpecs[timetype()].endValue);

    m_iov.markUpdate();
    return m_iov->add(sinceTime,payloadToken);

  }

 
  unsigned int 
 IOVEditorImpl::freeInsert( cond::Time_t sinceTime ,
			       const std::string& payloadToken
			       ){
    // reportError("cond::IOVEditorImpl::freeInsert not supported yet");

    if( m_token.empty() ) {
      reportError("cond::IOVEditorImpl::freeInsert cannot append to non-existing IOV index");
    }
    
    if(!m_isActive) {
      this->init();
    }
    
    //   if( m_iov->iov.empty() ) reportError("cond::IOVEditorImpl::freeInsert cannot insert  to empty IOV index");
    

   if(!validTime(sinceTime))
     reportError("cond::IOVEditorImpl::freeInsert time not in global range",sinceTime);

   
   // we do not support multiple iov with identical since...
   if (m_iov->exist(sinceTime))
     reportError("cond::IOVEditorImpl::freeInsert sinceTime already existing",sinceTime);



     // does it make sense? (in case of mixed till and since insertions...)
    if (lastTill()<sinceTime) updateClosure(timeTypeSpecs[timetype()].endValue);

    m_iov.markUpdate();
    return m_iov->add(sinceTime,payloadToken);
    
  }


  // remove last entry
  unsigned int IOVEditorImpl::truncate(bool withPayload) {
    if( m_token.empty() ) reportError("cond::IOVEditorImpl::truncate cannot delete to non-existing IOV sequence");
    if(!m_isActive) this->init();
    if (m_iov->piovs().empty()) return 0;
    if(withPayload){
      std::string tokenStr = m_iov->piovs().back().wrapperToken();
      m_poolDb.deleteObject( tokenStr );
    }
    m_iov.markUpdate();
    return m_iov->truncate();
    
  }


  void 
  IOVEditorImpl::deleteEntries(bool withPayload){
    if( m_token.empty() ) reportError("cond::IOVEditorImpl::deleteEntries cannot delete to non-existing IOV sequence");
    if(!m_isActive) this->init();
    if(withPayload){
      std::string tokenStr;
      IOVSequence::const_iterator payloadIt;
      IOVSequence::const_iterator payloadItEnd=m_iov->piovs().end();
      for(payloadIt=m_iov->piovs().begin();payloadIt!=payloadItEnd;++payloadIt){
        tokenStr=payloadIt->wrapperToken();
        m_poolDb.deleteObject( tokenStr );
      }
    }
    m_iov.markDelete();
  }

  void 
  IOVEditorImpl::import( const std::string& sourceIOVtoken ){
    if( !m_token.empty() ) reportError("cond::IOVEditorImpl::import IOV sequence already exists, cannot import");
    m_iov = m_poolDb.getTypedObject<cond::IOVSequence>(sourceIOVtoken);
    m_token=m_iov.toString();
  }
  

}
