#include "CondCore/DBCommon/interface/GenericRef.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "IOVEditorImpl.h"
#include "IOV.h"
#include "POOLCore/Token.h"
#include "StorageSvc/DbReflex.h"

namespace cond {

  IOVEditorImpl::IOVEditorImpl( cond::PoolTransaction& pooldb,
				const std::string& token
				):m_pooldb(&pooldb),m_token(token),
				  m_isActive(false){
  }
  
  void IOVEditorImpl::create( cond::Time_t firstSince,
			      cond::TimeType timetype) {

    if(!m_token.empty()){
      // problem??
      throw cond::Exception("cond::IOVEditorImpl::create cannot create a IOV using an initialized Editor");
    }

    if(!validTime(firstSince, timetype))
      throw cond::Exception("cond::IOVEditorImpl::create time not in global range");
      

    m_iov=cond::TypedRef<cond::IOV>(*m_pooldb,new cond::IOV);
    m_iov->timetype=(int)timetype;
    m_iov->firstsince=firstSince;
    
    m_iov.markWrite(cond::IOVNames::container());
    m_token=m_iov.token();
    m_isActive=true;
 
  }
 



  void IOVEditorImpl::init(){
    if(m_token.empty()){
      // problem?
      throw cond::Exception("cond::IOVEditorImpl::init cannot init w/o token change");
      
    }
    
    m_iov=cond::TypedRef<cond::IOV>(*m_pooldb, m_token); 
    m_isActive=true;
    
}



  IOVEditorImpl::~IOVEditorImpl(){
  }
  
  Time_t IOVEditorImpl::firstSince() const {
    return m_iov->firstsince;
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


    if(!validTime(tillTime))
      throw cond::Exception("cond::IOVEditorImpl::insert time not in global range");
  
    if(tillTime<firstSince() ||
       ( !m_iov->iov.empty() && tillTime<=m_iov->iov.back().first) 
       )    throw cond::Exception("cond::IOVEditorImpl::insert IOV not in range");
 
    m_iov.markUpdate();
    return m_iov->add(tillTime, payloadToken);
    
  }
  
  void 
  IOVEditorImpl::bulkInsert(std::vector< std::pair<cond::Time_t,std::string> >& values){
    if (values.empty()) return;
    if(!m_isActive) this->init();
    cond::Time_t tillTime = values.front().first;

    if(!validTime(tillTime))
      throw cond::Exception("cond::IOVEditorImpl::bulkInsert first time not in global range");

    if(!validTime(values.back().first))
      throw cond::Exception("cond::IOVEditorImpl::bulkInsert last time not in global range");

   if(tillTime<firstSince() ||
       ( !m_iov->iov.empty() && tillTime<=m_iov->iov.back().first) 
       )    throw cond::Exception("cond::IOVEditorImpl::bulkInsert IOV not in range");
    
    m_iov->iov.insert(m_iov->iov.end(), values.begin(), values.end());
    m_iov.markUpdate();   
  }


  void 
  IOVEditorImpl::updateClosure( cond::Time_t newtillTime ){
    if( m_token.empty() ) throw cond::Exception("cond::IOVEditorImpl::updateClosure cannot change non-existing IOV index");
    if(!m_isActive) this->init();
    m_iov->iov.back().first=newtillTime;
    m_iov.markUpdate();
  }
  
  unsigned int 
  IOVEditorImpl::append( cond::Time_t sinceTime ,
			       const std::string& payloadToken
			       ){
    if( m_token.empty() ) {
      throw cond::Exception("cond::IOVEditorImpl::appendIOV cannot append to non-existing IOV index");
    }
    
    if(!m_isActive) {
      this->init();
    }
    
    if( m_iov->iov.empty() ) throw cond::Exception("cond::IOVEditorImpl::appendIOV cannot append to empty IOV index");
    

   if(!validTime(sinceTime))
      throw cond::Exception("cond::IOVEditorImpl::append time not in global range");

   if (sinceTime<=firstSince())  throw cond::Exception("IOVEditor::append Error: since time out of range, below first since");
    
    
    if(  m_iov->iov.size()>1 ){
      //range check in case 
      cond::Time_t lastValidTill=(++m_iov->iov.rbegin())->first;
      //std::cout<<"lastValidTill "<<lastValidTill<<std::endl;
      if( (sinceTime-1)<= lastValidTill){
	throw cond::Exception("IOVEditor::append Error: since time out of range: below last since");
      }
    }

    cond::Time_t lastIOV=m_iov->iov.back().first;
    // does it make sense? (in case of mixed till and since insertions...)
    if (lastIOV<sinceTime) lastIOV=timeTypeSpecs[timetype()].endValue;
    m_iov.markUpdate();
    m_iov->iov.back().first = sinceTime-1;
    return m_iov->add(lastIOV,payloadToken);


  }

 unsigned int 
 IOVEditorImpl::freeInsert( cond::Time_t sinceTime ,
			       const std::string& payloadToken
			       ){
    if( m_token.empty() ) {
      throw cond::Exception("cond::IOVEditorImpl::appendIOV cannot append to non-existing IOV index");
    }
    
    if(!m_isActive) {
      this->init();
    }
    
    if( m_iov->iov.empty() ) throw cond::Exception("cond::IOVEditorImpl::appendIOV cannot insert  to empty IOV index");
    

   if(!validTime(sinceTime))
      throw cond::Exception("cond::IOVEditorImpl::freeInsert time not in global range");


 
   if (sinceTime<firstSince()) {
     m_iov->iov.insert(m_iov->iov.begin(),IOV::Item(firstSince()-1,payloadToken));
     m_iov->firstsince=sinceTime;
     m_iov.markUpdate();
    return 0;
   }

   // insert after found one.
   cond::Time_t tillTime;
   IOV::iterator p = m_iov->find(sinceTime);
   if (p==m_iov->iov.end()) {
     // closed range???
     tillTime=timeTypeSpecs[timetype()].endValue;
     (*(p-1)).first=sinceTime-1;

   }
   else {

     {
       // check for existing since
       if (p==m_iov->iov.begin() ) {
	 if (firstSince()==sinceTime)
	   throw cond::Exception("cond::IOVEditorImpl::freeInsert sinceTime already existing");
       } else
	 if ((*(p-1)).first==sinceTime-1)
	   throw cond::Exception("cond::IOVEditorImpl::freeInsert sinceTime already existing");
     }

     tillTime=(*p).first;
     (*p).first=sinceTime-1;
     p++;
   }

   p = m_iov->iov.insert(p,IOV::Item(tillTime,payloadToken));
   m_iov.markUpdate();
   return p - m_iov->iov.begin();

    
  }


  unsigned int 
  IOVEditorImpl::replaceInterval(cond::Time_t sinceTime,
				 cond::Time_t tillTime,
				 const std::string& payloadToken,
				 bool deletePayload) {

    static const std::string invalidToken(":");

    if( m_token.empty() ) {
     throw cond::Exception("cond::IOVEditorImpl::replaceInterval cannot edit an non-existing IOV index");
    }
    
    if(!m_isActive) {
      this->init();
    }
    
    if( m_iov->iov.empty() ) throw cond::Exception("cond::IOVEditorImpl::replaceInterval cannot replace in an empty IOV index");
    

   if(!validTime(sinceTime)||!validTime(tillTime)||tillTime<sinceTime)
     throw cond::Exception("cond::IOVEditorImpl::replaceInterval times not in IOVs range");

   {
     IOV::iterator b = m_iov->find(sinceTime);
     IOV::iterator e = m_iov->find(tillTime);
     // start cleaning
     if (b==m_iov->iov.begin()) {
       if (sinceTime>firstSince()) b++;
     } else if( b!=m_iov->iov.end() && sinceTime > (*(b-1)).first+1) b++;
     if (e!=m_iov->iov.end() && (*e).first==tillTime) e++;
     if (e>b) {
       if(deletePayload) {
	 for ( IOV::iterator p=b; p!=e; p++) {
	   cond::GenericRef ref(*m_pooldb,(*p).second);
	   ref.markDelete();
	   ref.reset();
	 }
       }
       m_iov->iov.erase(b,e);
     }
   }
   IOV::iterator p = m_iov->find(sinceTime);
   IOV::iterator e = m_iov->find(tillTime);
   if (e-p>1) 
     throw cond::Exception("cond::IOVEditorImpl::replaceInterval vincenzo logic has a fault!!!!");

   cond::Time_t oldTill;

   if (sinceTime<firstSince()) {
     oldTill = firstSince()-1;
     m_iov->firstsince=sinceTime;
   }else {
     if (p!=m_iov->iov.end()) p++;
     else {
       // pad....
       if (m_iov->iov.back().first<sinceTime-1)
	 p=m_iov->iov.insert(p,IOV::Item(sinceTime-1,invalidToken));
     }
     oldTill =  (*(p-1)).first;
     (*(p-1)).first=sinceTime-1;
   }
   p = m_iov->iov.insert(p,IOV::Item(tillTime,payloadToken));
   if (oldTill>tillTime) // split old interval
     p=m_iov->iov.insert(p+1,IOV::Item(oldTill,(*(p-1)).second));
   m_iov.markUpdate();
   return p - m_iov->iov.begin();

  }


  // delete entry at a given time
  unsigned int 
  IOVEditorImpl::deleteEntry(cond::Time_t time,
			       bool withPayload) {
   if( m_token.empty() ) {
      throw cond::Exception("cond::IOVEditorImpl::deleteEntry cannot delete from non-existing IOV index");
    }
    
    if(!m_isActive) {
      this->init();
    }
    
    if( m_iov->iov.empty() ) throw cond::Exception("cond::IOVEditorImpl::deleteEntry cannot delete from empty IOV index");
    

   if(!validTime(time)||time<firstSince())
     throw cond::Exception("cond::IOVEditorImpl::deleteEntry time not in IOVs range");

   IOV::iterator p = m_iov->find(time);
   if (p==m_iov->iov.end())
     throw cond::Exception("cond::IOVEditorImpl::deleteEntry time not in IOVs range");

   int n = p-m_iov->iov.begin();
   if(withPayload) {
     cond::GenericRef ref(*m_pooldb,(*p).second);
     ref.markDelete();
   }
   
   m_iov.markUpdate();
   if (p==m_iov->iov.begin() )
     m_iov->firstsince=(*p).first+1;
   else 
     (*(p-1)).first=(*p).first;
   m_iov->iov.erase(p);

   return n;

  }


  void 
  IOVEditorImpl::deleteEntries(bool withPayload){
    if( m_token.empty() ) throw cond::Exception("cond::IOVEditorImpl::deleteEntries cannot delete to non-existing IOV index");
    if(!m_isActive) this->init();
    if(withPayload){
      std::string tokenStr;
      IOV::iterator payloadIt;
      IOV::iterator payloadItEnd=m_iov->iov.end();
      for(payloadIt=m_iov->iov.begin();payloadIt!=payloadItEnd;++payloadIt){
	tokenStr=payloadIt->second;
	cond::GenericRef ref(*m_pooldb,tokenStr);
	ref.markDelete();
	ref.reset();
      }
    }
    m_iov.markDelete();
  }

  void 
  IOVEditorImpl::import( const std::string& sourceIOVtoken ){
    if( !m_token.empty() ) throw cond::Exception("cond::IOVEditorImpl::import IOV index already exists, cannot import");
    m_iov=cond::TypedRef<cond::IOV>(*m_pooldb,sourceIOVtoken);
    m_iov.markWrite(cond::IOVNames::container());
    m_token=m_iov.token();
  }
  

}
