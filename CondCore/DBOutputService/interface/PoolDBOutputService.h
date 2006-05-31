#ifndef CondCore_PoolDBOutputService_h
#define CondCore_PoolDBOutputService_h
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "CondCore/DBCommon/interface/DBWriter.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/IOVService/interface/IOV.h"
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
      void  postBeginJob();
      void  postEndJob();
      //no use
      //void preSource();
      //void postSource();
      //
      //use these to control transaction interval
      //
      void preEventProcessing( const edm::EventID & evtID, 
			       const edm::Timestamp & iTime );
      void postEventProcessing( const edm::Event & evt, 
				const edm::EventSetup & iEvtSetUp);
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
				       unsigned long long tillTime){
	std::string payloadTok=m_payloadWriter->markWrite(payloadObj);
	if( m_appendIOV ){
	  std::map<unsigned long long, std::string>::iterator 
	    lastIOVit=m_iov->iov.lower_bound(m_endOfTime);
	  unsigned long long closeIOVval=lastIOVit->first;
	  m_iov->iov.insert( std::make_pair(tillTime,lastIOVit->second) );
	  if( closeIOVval <= tillTime ){
	    throw cond::Exception(std::string("PoolDBOutputService::newValidityForNewPayload cannot append beyond IOV boundary"));
	  }else{
	    m_iov->iov[m_endOfTime]=payloadTok;
	  }
	}else{
	  m_iov->iov.insert(std::make_pair(tillTime,payloadTok));
	}
      }
      //
      // Service callback method
      // assign new validity to an existing payload object
      //
      void newValidityForOldPayload( const std::string& payloadObjToken,
				     unsigned long long tillTime );
      //
      // Service time utility callback method 
      // return the infinity value according to the given timetype
      // It is the IOV closing boundary
      //
      unsigned long long endOfTime();
      //
      // Service time utility callback method 
      // return the current conditions time value according to the 
      // given timetype
      //
      unsigned long long currentTime();
      virtual ~PoolDBOutputService();
    private:
      //
      // establish db connection, set up initial conditions
      //
      void connect();    
      void disconnect();
      std::string m_connect;
      std::string m_tag;
      std::string m_timetype;
      unsigned int m_connectMode;
      std::string m_containerName;
      std::string m_customMappingFile;
      bool m_appendIOV;
      std::string m_catalog;
      cond::ServiceLoader* m_loader;
      cond::MetaData* m_metadata;
      cond::IOV* m_iov;
      cond::DBSession* m_session;
      cond::DBWriter* m_payloadWriter;
      cond::DBWriter* m_iovWriter;
      unsigned long long m_endOfTime;
      unsigned long long m_currentTime;
      std::string m_iovToken; //iov token cache
    };//PoolDBOutputService
  }//ns service
}//ns cond
#endif
