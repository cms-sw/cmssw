#ifndef CondCore_PoolDBOutputService_h
#define CondCore_PoolDBOutputService_h
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
//#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "serviceCallbackRecord.h"
#include <string>
#include <map>
namespace edm{
  class Event;
  class EventSetup;
  class ParameterSet;
}
namespace cond{
  namespace service {
    class serviceCallbackToken;
    /** transaction and data consistency
	for create new tag, 
	start write metadata transaction only if the first pool commit 
	successful;
	for append,start readonly metadata transaction. start pool transaction only if metadata transaction successful.
	
    */
    class PoolDBOutputService{
    public:
      PoolDBOutputService( const edm::ParameterSet & iConfig, 
			   edm::ActivityRegistry & iAR );
      //use these to control connections
      //void  postBeginJob();
      void  postEndJob();
      //
      //use these to control transaction interval
      //
      void preEventProcessing( const edm::EventID & evtID, 
      			       const edm::Timestamp & iTime );
      void preModule(const edm::ModuleDescription& desc);
      void postModule(const edm::ModuleDescription& desc);
      //
      // return the database session in use
      //
      cond::DBSession& session() const;
      std::string tag( const std::string& EventSetupRecordName );
      bool isNewTagRequest( const std::string& EventSetupRecordName );
      //
      // insert the payload and its valid till time into the database
      // Note: user looses the ownership of the pointer to the payloadObj
      // The payload object will be stored as well
      // 
      template<typename T>
	void createNewIOV( T* firstPayloadObj, 
			   cond::Time_t firstTillTime,
			   const std::string& EventSetupRecordName
			   ){
	cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
	if ( !m_dbstarted ) {
	  this->initDB();
	}
	if(!myrecord.m_isNewTag) throw cond::Exception("not a new tag");
	cond::PoolTransaction& pooldb=m_connection->poolTransaction(false);
	std::string iovToken;
	try{
	  pooldb.start();
	  cond::TypedRef<T> myPayload(pooldb,firstPayloadObj);
	  myPayload.markWrite(EventSetupRecordName);
	  std::string payloadToken=myPayload.token();
	  iovToken=this->insertIOV(myrecord,payloadToken,firstTillTime,EventSetupRecordName);
	  pooldb.commit();
	}catch(...){
	  pooldb.rollback();
	  throw cond::Exception("createNewIOV: error in commit in pool");
	}
	cond::CoralTransaction& coraldb=m_connection->coralTransaction(false);
	try{
	  cond::MetaData metadata(coraldb);
	  coraldb.start();
	  metadata.addMapping(myrecord.m_tag,iovToken);
	  coraldb.commit();
	}catch(...){
	  //I hope it'll never happen!
	  coraldb.rollback();
	  //delete new payload; 
	  pooldb.start();
	  //this->resetPreviousPoolStatus(payloadToken,iovToken);
	  pooldb.commit();
	}
	m_newtags.push_back( std::make_pair<std::string,std::string>(myrecord.m_tag,iovToken) );
	myrecord.m_isNewTag=false;
      }
      void createNewIOV( const std::string& firstPayloadToken, 
			cond::Time_t firstTillTime,
			const std::string& EventSetupRecordName );
      template<typename T>
	void appendTillTime( T* payloadObj, 
			     cond::Time_t tillTime,
			     const std::string& EventSetupRecordName
			     ){
	cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
	if (!m_dbstarted) this->initDB();
	cond::PoolTransaction& pooldb=m_connection->poolTransaction(false);
	try{
	  pooldb.start();
	  cond::TypedRef<T> myPayload(pooldb,payloadObj);
	  myPayload.markWrite(EventSetupRecordName);
	  std::string payloadToken=myPayload.token();
	  pooldb.commit();
	  std::string iovToken=this->insertIOV(myrecord,payloadToken,tillTime,EventSetupRecordName);
	}catch(...){
	  pooldb.rollback();
	}
      }
      void appendTillTime( const std::string& payloadToken, 
			   cond::Time_t tillTime,
			   const std::string& EventSetupRecordName
			    );
      
      template<typename T>
	void appendSinceTime( T* payloadObj, 
			      cond::Time_t sinceTime,
			      const std::string& EventSetupRecordName ){
	cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
	if (!m_dbstarted) this->initDB();
	cond::PoolTransaction& pooldb=m_connection->poolTransaction(false);
	try{
	  pooldb.start();
	  cond::TypedRef<T> myPayload(pooldb,payloadObj);
	  myPayload.markWrite(EventSetupRecordName);
	  std::string payloadToken=myPayload.token();
	  this->appendIOV(myrecord,payloadToken,sinceTime);
	  pooldb.commit();
	}catch(...){
	  pooldb.rollback();
	}
      }
      //
      // Append the payload and its valid sinceTime into the database
      // Note: user looses the ownership of the pointer to the payloadObj
      // Note: the iov index appended to MUST pre-existing and the existing 
      // conditions data are retrieved from EventSetup 
      // 
      void appendSinceTime( const std::string& payloadToken, 
			   cond::Time_t sinceTime,
			   const std::string& EventSetupRecordName );

      //
      // Service time utility callback method 
      // return the infinity value according to the given timetype
      // It is the IOV closing boundary
      //
      cond::Time_t endOfTime() const;
      //
      // Service time utility callback method 
      // return the current conditions time value according to the 
      // given timetype
      //
      cond::Time_t currentTime() const;
      virtual ~PoolDBOutputService();
    private:
      void connect();    
      void disconnect();
      void initDB();
      size_t callbackToken(const std::string& EventSetupRecordName ) const ;
      void appendIOV(cond::service::serviceCallbackRecord& record,
		     const std::string& payloadToken, 
		     cond::Time_t sinceTime);
      std::string insertIOV(cond::service::serviceCallbackRecord& record,
			    const std::string& payloadToken, 
			    cond::Time_t tillTime, const std::string& EventSetupRecordName);
      serviceCallbackRecord& lookUpRecord(const std::string& EventSetupRecordName);
    private:
      cond::Time_t m_currentTime;
      cond::DBSession* m_session;
      cond::Connection* m_connection;
      std::map<size_t, cond::service::serviceCallbackRecord> m_callbacks;
      std::vector< std::pair<std::string,std::string> > m_newtags;
      bool m_dbstarted;
      cond::IOVService* m_iovservice;
      //edm::ParameterSet m_connectionPset;
    };//PoolDBOutputService
  }//ns service
}//ns cond
#endif
