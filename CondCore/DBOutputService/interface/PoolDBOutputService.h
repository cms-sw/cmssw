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
  //class IOV;
  namespace service {
    class PoolDBOutputService{
    public:
      //
      // accepted PSet 
      // moduleToWatch
      // connect, connectMode, 
      // authenticationMethod, containerName,payloadCustomMappingFile
      // commitInterval, appendIOV, catalog,tag,
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
      //
      //
      void preModule ( const edm::ModuleDescription & iDesc );
      void postModule( const edm::ModuleDescription & iDesc );
      //no use
      void preModuleConstruction ( const edm::ModuleDescription & iDesc );
      void postModuleConstruction ( const edm::ModuleDescription & iDesc );
      //no use
      //void preSourceConstruction ( const edm::ModuleDescription& );
      //void postSourceConstruction ( const edm::ModuleDescription& );
      //
      // callback method
      // assign validity to a new payload object
      // if in append mode, the service reassign the previous tilltime to 
      // the current time and assign the current tilltime as
      //
      template<typename T>
	void newValidityForNewPayload( T* payloadObj, 
				       unsigned long long tillTime){
	std::string payloadTok=m_payloadWriter->markWrite(payloadObj);
	if( m_appendIOV ){
	  std::map<unsigned long long, std::string>::iterator 
	    lastIOVit = (m_iov->iov.end())--;
	  ////if(tillTime < m_currentTime ) throw exception
	  m_iov->iov.insert( std::make_pair(m_currentTime,lastIOVit->second) );
	  m_iov->iov.erase(lastIOVit);
	}
	m_iov->iov.insert(std::make_pair(tillTime,payloadTok));
      }
      //
      // callback method
      // assign new validity to an existing payload object
      //
      void newValidityForOldPayload( const std::string& payloadObjToken,
				     unsigned long long tillTime );
      //time utilities
      unsigned long long endOfTime();
      unsigned long long currentTime();
      virtual ~PoolDBOutputService();
    private:
      //
      // establish db connection, set up initial conditions
      //
      void connect();    
      void disconnect();
      void flushIOV();
      void appendIOV();
      std::string m_connect;
      std::string m_tag;
      std::string m_timetype;
      std::string m_clientmodule; //only one client is allowed for the moment
      unsigned int m_connectMode;
      //unsigned int m_authenticationMethod;
      std::string m_containerName;
      std::string m_customMappingFile;
      unsigned int m_commitInterval; //the interval is per object
      bool m_appendIOV;
      std::string m_catalog;
      //bool m_loadBlobStreamer;
      //unsigned int m_messageLevel;
      cond::ServiceLoader* m_loader;
      cond::MetaData* m_metadata;
      cond::IOV* m_iov;
      cond::DBSession* m_session;
      cond::DBWriter* m_payloadWriter;
      bool m_transactionOn;
      unsigned long long m_endOfTime;
      unsigned long long m_currentTime;
    };//PoolDBOutputService
  }//ns service
}//ns cond
#endif
