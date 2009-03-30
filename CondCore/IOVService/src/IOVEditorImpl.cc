#include "CondCore/DBCommon/interface/GenericRef.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "IOVEditorImpl.h"
#include "CondFormats/Common/interface/IOVSequence.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbReflex.h"

namespace cond {

  IOVEditorImpl::IOVEditorImpl( cond::PoolTransaction& pooldb,
				const std::string& token
				):m_pooldb(&pooldb),m_token(token),
				  m_isActive(false){
  }
  
  void IOVEditorImpl::create(cond::TimeType timetype,cond::Time_t lastTill) {

    if(!m_token.empty()){
      // problem??
      throw cond::Exception("cond::IOVEditorImpl::create cannot create a IOV using an initialized Editor");
    }

    if(!validTime(lastTill, timetype))
      throw cond::Exception("cond::IOVEditorImpl::create time not in global range");
    
    
    m_iov=cond::TypedRef<cond::IOVSequence>(*m_pooldb,new cond::IOVSequence((int)timetype,lastTill," "));
					    
    m_iov.markWrite(cond::IOVNames::container());
    m_token=m_iov.token();
    m_isActive=true;
    
  }
  


  
  void IOVEditorImpl::init(){
    if(m_token.empty()){
      // problem?
      throw cond::Exception("cond::IOVEditorImpl::init cannot init w/o token change");
      
    }
    
    m_iov=cond::TypedRef<cond::IOVSequence>(*m_pooldb, m_token); 
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
    
    if( m_iov->iovs().empty() ) throw cond::Exception("cond::IOVEditorImpl::insert cannot inser into empty IOV sequence");
    
    if(!validTime(tillTime))
      throw cond::Exception("cond::IOVEditorImpl::insert time not in global range");
    
    if(tillTime<=lastTill() )
      throw cond::Exception("cond::IOVEditorImpl::insert IOV not in range");
    
    m_iov.markUpdate();
    updateClosure(tillTime);
    return m_iov->add(lastTill(), payloadToken);
    
  }
  
  void 
  IOVEditorImpl::bulkAppend(std::vector< std::pair<cond::Time_t,std::string> >& values){
    if (values.empty()) return;
    if(!m_isActive) this->init();
    cond::Time_t firstTime = values.front().first;
    cond::Time_t  lastTime = values.back().first;

    if(!validTime(firstTime))
      throw cond::Exception("cond::IOVEditorImpl::bulkInsert first time not in global range");

    if(!validTime(lastTime))
      throw cond::Exception("cond::IOVEditorImpl::bulkInsert last time not in global range");

   if(lastTime>=lastTill() ||
      ( !m_iov->iovs().empty() && firstTime<=m_iov->iovs().back().sinceTime()) 
       )    throw cond::Exception("cond::IOVEditorImpl::bulkInsert IOV not in range");
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
      throw cond::Exception("cond::IOVEditorImpl::bulkInsert first time not in global range");

    if(!validTime(lastTime))
      throw cond::Exception("cond::IOVEditorImpl::bulkInsert last time not in global range");

   if(lastTime>=lastTill() ||
      ( !m_iov->iovs().empty() && firstTime<=m_iov->iovs().back().sinceTime()) 
       )    throw cond::Exception("cond::IOVEditorImpl::bulkInsert IOV not in range");

   // m_iov->iov.insert(m_iov->iov.end(), values.begin(), values.end());
   m_iov.markUpdate();   
 
 }
  

  void 
  IOVEditorImpl::updateClosure( cond::Time_t newtillTime ){
    if( m_token.empty() ) throw cond::Exception("cond::IOVEditorImpl::updateClosure cannot change non-existing IOV index");
    if(!m_isActive) this->init();
    m_iov->updateLastTill(newtillTime);
    m_iov.markUpdate();
  }
  
  unsigned int 
  IOVEditorImpl::append( cond::Time_t sinceTime,
			 const std::string&payloadToken
			 ){
    if( m_token.empty() ) {
      throw cond::Exception("cond::IOVEditorImpl::appendIOV cannot append to non-existing IOV index");
    }
    
    if(!m_isActive) {
      this->init();
    }

    if(!validTime(sinceTime))
      throw cond::Exception("cond::IOVEditorImpl::append time not in global range");
    
    
    if(  m_iov->iovs().size()>1 ){
      //range check in case 
      cond::Time_t lastValidSince=m_iov->iovs().back().sinceTime();
      //std::cout<<"lastValidTill "<<lastValidTill<<std::endl;
      if( sinceTime<= lastValidSince){
	throw cond::Exception("IOVEditor::append Error: since time out of range: below last since");
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
    // throw cond::Exception("cond::IOVEditorImpl::freeInsert not supported yet");

    if( m_token.empty() ) {
      throw cond::Exception("cond::IOVEditorImpl::freeInsert cannot append to non-existing IOV index");
    }
    
    if(!m_isActive) {
      this->init();
    }
    
    //   if( m_iov->iov.empty() ) throw cond::Exception("cond::IOVEditorImpl::freeInsert cannot insert  to empty IOV index");
    

   if(!validTime(sinceTime))
      throw cond::Exception("cond::IOVEditorImpl::freeInsert time not in global range");


   IOVSequence::const_iterator p = m_iov->find(sinceTime);
   if (p!=m_iov->iovs().end() &&  (*p).sinceTime()==sinceTime)
     throw cond::Exception("cond::IOVEditorImpl::freeInsert sinceTime already existing");



     // does it make sense? (in case of mixed till and since insertions...)
    if (lastTill()<sinceTime) updateClosure(timeTypeSpecs[timetype()].endValue);

    m_iov.markUpdate();
    return m_iov->add(sinceTime,payloadToken);
    
  }



  void 
  IOVEditorImpl::deleteEntries(bool withPayload){
    if( m_token.empty() ) throw cond::Exception("cond::IOVEditorImpl::deleteEntries cannot delete to non-existing IOV sequence");
    if(!m_isActive) this->init();
    if(withPayload){
      std::string tokenStr;
      IOVSequence::const_iterator payloadIt;
      IOVSequence::const_iterator payloadItEnd=m_iov->iovs().end();
      for(payloadIt=m_iov->iovs().begin();payloadIt!=payloadItEnd;++payloadIt){
	tokenStr=payloadIt->wrapperToken();
	cond::GenericRef ref(*m_pooldb,tokenStr);
	ref.markDelete();
	ref.reset();
      }
    }
    m_iov.markDelete();
  }

  void 
  IOVEditorImpl::import( const std::string& sourceIOVtoken ){
    if( !m_token.empty() ) throw cond::Exception("cond::IOVEditorImpl::import IOV sequence already exists, cannot import");
    m_iov=cond::TypedRef<cond::IOVSequence>(*m_pooldb,sourceIOVtoken);
    m_iov.markWrite(cond::IOVNames::container());
    m_token=m_iov.token();
  }
  

}
