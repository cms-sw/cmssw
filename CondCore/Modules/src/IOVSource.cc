#include "CondCore/Modules/src/IOVSource.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/Framework/interface/Event.h"
#include "DataFormats/Common/interface/EventID.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ServiceLoader.h"
#include "CondCore/IOVService/interface/IOV.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "DataSvc/Ref.h"
#include <iostream>
namespace cond{
  //allowed parameters: firstRun, firstTime, lastRun, lastTime, 
  //common paras: connect, catalog, authenticationMethod, timetype,messagelevel
  IOVSource::IOVSource(edm::ParameterSet const& pset,
		       edm::InputSourceDescription const& desc):
    edm::ConfigurableInputSource(pset,desc),
    m_loader(new cond::ServiceLoader),
    m_connect(pset.getParameter<std::string>("connect")),
    m_tag(pset.getParameter<std::string>("tag")),
    m_catconnect(pset.getUntrackedParameter<std::string>("catalog","")),
    m_timeType(pset.getParameter<std::string>("timetype")){
    unsigned int auth=pset.getUntrackedParameter<unsigned int>("authenticationMethod",0) ;
    unsigned int message_level=pset.getUntrackedParameter<unsigned int>("messagelevel",0);
    try{
      if( auth==1 ){
	m_loader->loadAuthenticationService( cond::XML );
      }else{
	m_loader->loadAuthenticationService( cond::Env );
      }
      switch (message_level) {
      case 0 :
	m_loader->loadMessageService(cond::Error);
	break;    
      case 1:
	m_loader->loadMessageService(cond::Warning);
	break;
      case 2:
	m_loader->loadMessageService( cond::Info );
	break;
      case 3:
	m_loader->loadMessageService( cond::Debug );
	break;  
      default:
	m_loader->loadMessageService();
      }
    }catch( const cond::Exception& e){
      throw e;
    }catch(const cms::Exception&e ){
      throw e;
    }catch ( ... ) {
      throw cond::Exception("IOVSource::IOVSource unknown error ");
    }
    this->getIOV();
    //for(unsigned long long i=1; i<10; i+=2){
    //  m_iovs.insert(i);//fake iov
    //}
    if( m_timeType=="runnumber" ){
      m_firstValid=pset.getUntrackedParameter<unsigned int>("firstRun",0);
      m_lastValid=(unsigned long long)pset.getUntrackedParameter<unsigned int>("lastRun",0);
    }else{
      m_firstValid=pset.getUntrackedParameter<unsigned int>("firstTime",0);
      m_lastValid=(unsigned long long)pset.getUntrackedParameter<unsigned int>("lastTime",0);
    }
    if(m_firstValid==0){
      m_iovit=m_iovs.begin();
    }else{
      std::set<unsigned long long>::iterator startpos=m_iovs.lower_bound((unsigned long long)m_firstValid);
      m_iovit=startpos;
    }
    if(m_lastValid==0){
      m_iovstop=m_iovs.end();
    }else{
      std::set<unsigned long long>::iterator stoppos=m_iovs.upper_bound((unsigned long long)m_lastValid);
      m_iovstop=stoppos;
    }
  }
  IOVSource::~IOVSource() {
    delete m_loader;
  }
  bool IOVSource::produce( edm::Event & e ) {
    if( (m_iovit==m_iovstop) ){
      return false;
    }
    m_iovit++; 
    return true;
  }  
  void IOVSource::setRunAndEventInfo(){
    if(m_iovit==m_iovs.end()) return;
    m_currentValid=*m_iovit;
    if( m_timeType=="runnumber" ){
      setRunNumber(m_currentValid);
    }else{
      setTime(m_currentValid);
    }
    setEventNumber(1); 
  }
  void IOVSource::getIOV(){
    try{
      cond::ServiceLoader* loader=new cond::ServiceLoader;
      cond::MetaData meta(m_connect, *loader);
      meta.connect( cond::ReadOnly );
      std::string iovToken=meta.getToken(m_tag);
      meta.disconnect();
      if( iovToken.empty() ){
	throw cond::Exception("IOVSource::getIOV empty tag");
      }
      cond::DBSession* session=new cond::DBSession(m_connect);
      session->setCatalog(m_catconnect);
      session->connect( cond::ReadOnly );
      session->startReadOnlyTransaction();
      pool::Ref<cond::IOV> iov(&(session->DataSvc()), iovToken);
      for(std::map<unsigned long long, std::string>::iterator it=iov->iov.begin(); it!=iov->iov.end(); ++it){
	m_iovs.insert(it->first);
      }
      session->commit();
      session->disconnect();
      delete session;
    }catch(const cond::Exception&e ){
      throw e;
    }catch(const cms::Exception&e ){
      throw e;
    }catch(...){
      throw cond::Exception( "IOVSource::getIOV " );
    }
  }
}//ns cond
