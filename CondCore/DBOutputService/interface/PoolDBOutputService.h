#ifndef CondCore_PoolDBOutputService_h
#define CondCore_PoolDBOutputService_h
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "CondCore/DBCommon/interface/DBWriter.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/IOVService/interface/IOV.h"
#include "serviceCallbackRecord.h"
#include <string>
#include <map>
namespace edm{
  class Event;
  class EventSetup;
  class ParameterSet;
}

namespace cond{
  class ServiceLoader;
  class MetaData;
  namespace service {
    class serviceCallbackToken;
    class PoolDBOutputService{
    public:
      //
      // accepted PSet 
      // connect, connectMode, 
      // authenticationMethod, containerName,payloadCustomMappingFile
      // appendIOV, catalog,tag,
      // loadBlobStreamer
      //
      PoolDBOutputService( const edm::ParameterSet & iConfig, 
			   edm::ActivityRegistry & iAR );
      //use these to control connections
      //void  postBeginJob();
      void  postEndJob();
      //no use
      //void preSource();
      //void postSource();
      //
      //use these to control transaction interval
      //
      void preEventProcessing( const edm::EventID & evtID, 
      			       const edm::Timestamp & iTime );
      //void postEventProcessing( const edm::Event & evt, 
      //				const edm::EventSetup & iEvtSetUp);
      //
      // Service Callback method
      // assign validity to a new payload object
      // The meaning of the tillTime argument:
      // --if in append mode, it is the tillTime you assign to the last 
      //    payload object in the IOV sequence you append to. 
      //    If the tillTime argument value goes beyond the IOV 
      // closing boundary(currently, EndOfTime), an exception is thrown  
      // --if not in append mode, it is the tillTime you assign to the payload
      //   object given in the first argument
      // 
      template<typename T>
	void newValidityForNewPayload( T* payloadObj, 
				       unsigned long long tillTime,
				       size_t callbackToken){
	std::map<size_t,cond::service::serviceCallbackRecord>::iterator it=m_callbacks.find(callbackToken);
	if(it==m_callbacks.end()) throw cond::Exception(std::string("PoolDBOutputService::newValidityForNewPayload: unregistered callback token"));
	cond::service::serviceCallbackRecord& myrecord=it->second;
	if (!m_dbstarted) this->initDB();
	if(!myrecord.m_payloadWriter){
	  myrecord.m_payloadWriter=new cond::DBWriter(*m_session,myrecord.m_containerName);
	}
	std::string payloadTok=myrecord.m_payloadWriter->markWrite(payloadObj);
	if( myrecord.m_appendIOV ){
	  std::map<unsigned long long, std::string>::iterator 
	    lastIOVit=myrecord.m_iov->iov.lower_bound(m_endOfTime);
	  unsigned long long closeIOVval=lastIOVit->first;
	  myrecord.m_iov->iov.insert( std::make_pair(tillTime,lastIOVit->second) );
	  if( closeIOVval <= tillTime ){
	    throw cond::Exception(std::string("PoolDBOutputService::newValidityForNewPayload cannot append beyond IOV boundary"));
	  }
	  myrecord.m_iov->iov[m_endOfTime]=payloadTok;
	}else{
	  myrecord.m_iov->iov.insert(std::make_pair(tillTime,payloadTok));
	}
      }
      //
      // Service callback method
      // assign new validity to an existing payload object  
      //
      void newValidityForOldPayload( const std::string& payloadObjToken,
				     unsigned long long tillTime,
				     size_t callbackToken);
      //
      // Service time utility callback method 
      // return the infinity value according to the given timetype
      // It is the IOV closing boundary
      //
      unsigned long long endOfTime() const;
      //
      // Service time utility callback method 
      // return the current conditions time value according to the 
      // given timetype
      //
      unsigned long long currentTime() const;
      virtual ~PoolDBOutputService();
      size_t callbackToken(const std::string& containerName) const ;
    private:
      void connect();    
      void disconnect();
      void initDB();
    private:
      std::string m_connect;
      std::string m_timetype;
      unsigned int m_connectMode;
      //std::string m_customMappingFile;
      std::string m_catalog;
      unsigned long long m_endOfTime;
      unsigned long long m_currentTime;
      cond::ServiceLoader* m_loader;
      cond::MetaData* m_metadata;
      cond::DBSession* m_session;
      cond::DBWriter* m_iovWriter;
      std::map<size_t, cond::service::serviceCallbackRecord> m_callbacks;
      bool m_dbstarted;
    };//PoolDBOutputService
  }//ns service
}//ns cond
#endif
