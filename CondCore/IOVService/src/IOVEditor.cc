#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondFormats/Common/interface/IOVSequence.h"
//#include "POOLCore/Token.h"
//#include "StorageSvc/DbReflex.h"
#include<ostream>
#include<sstream>

namespace cond {

  IOVEditor::~IOVEditor(){}

  IOVEditor::IOVEditor( cond::DbSession& dbSess):m_dbSess(dbSess),
						 m_isActive(false){
  }

  IOVEditor::IOVEditor( cond::DbSession& dbSess,
			const std::string& token
			):m_dbSess(dbSess),m_token(token),
			  m_isActive(false){
  }
  
  IOVSequence & IOVEditor::iov(){ return *m_iov;}


  void IOVEditor::debugInfo(std::ostream & co) const {
    co << "IOVEditor: ";
    co << "db " << m_dbSess.connectionString();
    if(m_token.empty()) {
      co << " no token"; return;
    }
    if (!m_iov.get() )  {
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

  void IOVEditor::reportError(std::string message) const {
    std::ostringstream out;
    out << "Error in ";
    debugInfo(out);
    out  << "\n" << message;
    throw cond::Exception(out.str());
  }

  void IOVEditor::reportError(std::string message, cond::Time_t time) const {
    std::ostringstream out;
    out << "Error in";
    debugInfo(out);
    out << "\n" <<  message << " for time:  " << time;
    throw cond::Exception(out.str());
  }


  // create empty default sequence
  void IOVEditor::create(cond::TimeType timetype) {
    if(!m_token.empty()){
      // problem??
      reportError("cond::IOVEditor::create cannot create a IOV using an initialized Editor");
    }

    m_iov.reset( new cond::IOVSequence(timetype) );
    m_token = m_dbSess.storeObject( m_iov.get(), cond::IOVNames::container());
    m_isActive=true;

  }

  void IOVEditor::create(cond::TimeType timetype,cond::Time_t lastTill) {
    if(!m_token.empty()){
      // problem??
      reportError("cond::IOVEditor::create cannot create a IOV using an initialized Editor");
    }
    
    if(!validTime(lastTill, timetype))
      reportError("cond::IOVEditor::create time not in global range",lastTill);
    
    m_iov.reset( new cond::IOVSequence((int)timetype,lastTill," ") );
    m_token = m_dbSess.storeObject( m_iov.get(),cond::IOVNames::container());
    m_isActive=true;
  }
    
  void IOVEditor::init(){
    if(m_token.empty()){
      // problem?
      reportError("cond::IOVEditor::init cannot init w/o token change");
    }
    
    m_iov = m_dbSess.getTypedObject<cond::IOVSequence>(m_token);
    m_isActive=true;
    
  }
  
    
  Time_t IOVEditor::firstSince() const {
    return m_iov->firstSince();
  }
  
  Time_t IOVEditor::lastTill() const {
    return m_iov->lastTill();
  }
  
  TimeType IOVEditor::timetype() const {
    return m_iov->timeType();
  }
  
  
  bool IOVEditor::validTime(cond::Time_t time, cond::TimeType timetype) const {
    return time>=timeTypeSpecs[timetype].beginValue && time<=timeTypeSpecs[timetype].endValue;   
    
  }
  
  bool IOVEditor::validTime(cond::Time_t time) const {
    return validTime(time,timetype());
  }
  
  
  
  unsigned int
  IOVEditor::insert( cond::Time_t tillTime,
		     const std::string& payloadToken
		     ){
    if(!m_isActive) this->init();
    
    if( m_iov->iovs().empty() ) 
      reportError("cond::IOVEditor::insert cannot inser into empty IOV sequence",tillTime);
    
    if(!validTime(tillTime))
      reportError("cond::IOVEditor::insert time not in global range",tillTime);
    
    if(tillTime<=lastTill() )
      reportError("cond::IOVEditor::insert IOV not in range",tillTime);
    
    cond::Time_t newSince=lastTill()+1;
    unsigned int ret = m_iov->add(newSince, payloadToken);
    updateClosure(tillTime);
    m_dbSess.updateObject( m_iov.get(), m_token );
    return ret;
  }
  
  void 
  IOVEditor::bulkAppend(std::vector< std::pair<cond::Time_t,std::string> >& values){
    if (values.empty()) return;
    if(!m_isActive) this->init();
    cond::Time_t firstTime = values.front().first;
    cond::Time_t  lastTime = values.back().first;
    if(!validTime(firstTime))
      reportError("cond::IOVEditor::bulkInsert first time not in global range",firstTime);

    if(!validTime(lastTime))
      reportError("cond::IOVEditor::bulkInsert last time not in global range",lastTime);

   if(lastTime>=lastTill() ||
      ( !m_iov->iovs().empty() && firstTime<=m_iov->iovs().back().sinceTime()) 
       )    
     reportError("cond::IOVEditor::bulkInsert IOV not in range",firstTime);

   for(std::vector< std::pair<cond::Time_t,std::string> >::const_iterator it=values.begin(); it!=values.end(); ++it){
     //     m_iov->iov.insert(m_iov->iov.end(), values.begin(), values.end());
     m_iov->add(it->first,it->second);
   }
   m_dbSess.updateObject( m_iov.get(), m_token );
  }

  void 
  IOVEditor::bulkAppend(std::vector< cond::IOVElement >& values){
    if (values.empty()) return;
    if(!m_isActive) this->init();
    cond::Time_t firstTime = values.front().sinceTime();
    cond::Time_t   lastTime = values.back().sinceTime();
    if(!validTime(firstTime))
      reportError("cond::IOVEditor::bulkInsert first time not in global range",firstTime);

    if(!validTime(lastTime))
      reportError("cond::IOVEditor::bulkInsert last time not in global range",lastTime);

   if(lastTime>=lastTill() ||
      ( !m_iov->iovs().empty() && firstTime<=m_iov->iovs().back().sinceTime()) 
      )    reportError("cond::IOVEditor::bulkInsert IOV not in range",firstTime);

   m_dbSess.updateObject( m_iov.get(), m_token );
 }
  

  void 
  IOVEditor::stamp(std::string const & icomment, bool append) {
    if(!m_isActive) this->init();
    m_iov->stamp(icomment, append);
    m_dbSess.updateObject( m_iov.get(), m_token );
  }


  void 
  IOVEditor::updateClosure( cond::Time_t newtillTime ){
    if( m_token.empty() ) reportError("cond::IOVEditor::updateClosure cannot change non-existing IOV index");
    if(!m_isActive) this->init();
    m_iov->updateLastTill(newtillTime);
    m_dbSess.updateObject( m_iov.get(), m_token );
  }
  
  unsigned int 
  IOVEditor::append( cond::Time_t sinceTime,
		     const std::string&payloadToken
		     ){
    if( m_token.empty() ) {
      reportError("cond::IOVEditor::appendIOV cannot append to non-existing IOV index");
    }
    
    if(!m_isActive) {
      this->init();
    }

    if(!validTime(sinceTime))
      reportError("cond::IOVEditor::append time not in global range",sinceTime);
    
    
    if(  !m_iov->iovs().empty() ){
      //range check in case 
      cond::Time_t lastValidSince=m_iov->iovs().back().sinceTime();
      if( sinceTime<= lastValidSince){
	reportError("IOVEditor::append Error: since time out of range: below last since",sinceTime);
      }
    }

    // does it make sense? (in case of mixed till and since insertions...)
    if (lastTill()<=sinceTime) updateClosure(timeTypeSpecs[timetype()].endValue);
    unsigned int ret = m_iov->add(sinceTime,payloadToken);
    m_dbSess.updateObject( m_iov.get(), m_token );
    return ret;
  }

 
  unsigned int 
 IOVEditor::freeInsert( cond::Time_t sinceTime ,
			const std::string& payloadToken
			){
    // reportError("cond::IOVEditor::freeInsert not supported yet");
    
    if( m_token.empty() ) {
      reportError("cond::IOVEditor::freeInsert cannot append to non-existing IOV index");
    }
    
    if(!m_isActive) {
      this->init();
    }
    
    //   if( m_iov->iov.empty() ) reportError("cond::IOVEditor::freeInsert cannot insert  to empty IOV index");
    

   if(!validTime(sinceTime))
     reportError("cond::IOVEditor::freeInsert time not in global range",sinceTime);

   
   // we do not support multiple iov with identical since...
   if (m_iov->exist(sinceTime))
     reportError("cond::IOVEditor::freeInsert sinceTime already existing",sinceTime);



     // does it make sense? (in case of mixed till and since insertions...)
   if (lastTill()<sinceTime) updateClosure(timeTypeSpecs[timetype()].endValue);
   unsigned int ret = m_iov->add(sinceTime,payloadToken);
   
   m_dbSess.updateObject( m_iov.get(), m_token );
   return ret;
  }


  // remove last entry
  unsigned int IOVEditor::truncate(bool withPayload) {
    if( m_token.empty() ) reportError("cond::IOVEditor::truncate cannot delete to non-existing IOV sequence");
    if(!m_isActive) this->init();
    if (m_iov->piovs().empty()) return 0;
    if(withPayload){
      std::string tokenStr = m_iov->piovs().back().wrapperToken();
      m_dbSess.deleteObject( tokenStr );
    }
    unsigned int ret = m_iov->truncate();
    m_dbSess.updateObject( m_iov.get(), m_token );
    return ret;
    
  }


  void 
  IOVEditor::deleteEntries(bool withPayload){
    if( m_token.empty() ) reportError("cond::IOVEditor::deleteEntries cannot delete to non-existing IOV sequence");
    if(!m_isActive) this->init();
    if(withPayload){
      std::string tokenStr;
      IOVSequence::const_iterator payloadIt;
      IOVSequence::const_iterator payloadItEnd=m_iov->piovs().end();
      for(payloadIt=m_iov->piovs().begin();payloadIt!=payloadItEnd;++payloadIt){
        tokenStr=payloadIt->wrapperToken();
        m_dbSess.deleteObject( tokenStr );
      }
    }
    m_dbSess.deleteObject( m_token );
    m_iov->piovs().clear();
  }

  void 
  IOVEditor::import( const std::string& sourceIOVtoken ){
    if( !m_token.empty() ) reportError("cond::IOVEditor::import IOV sequence already exists, cannot import");
    m_iov = m_dbSess.getTypedObject<cond::IOVSequence>(sourceIOVtoken);
    //m_token=m_iov.toString();
    m_token=sourceIOVtoken;
  }
  

}
