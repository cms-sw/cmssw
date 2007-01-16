#ifndef CondCore_PoolDBOutputService_h
#define CondCore_PoolDBOutputService_h
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "CondCore/DBCommon/interface/DBSession.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/DBCommon/interface/PoolStorageManager.h"
#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "CondCore/DBCommon/interface/Ref.h"
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
	if (!m_dbstarted) this->initDB();
	m_pooldb->connect();
	m_pooldb->startTransaction(false);    
	cond::Ref<T> myPayload(*m_pooldb,firstPayloadObj);
	myPayload.markWrite(EventSetupRecordName);
	std::string payloadToken=myPayload.token();
	std::string iovToken=this->insertIOV(myrecord,payloadToken,firstTillTime,EventSetupRecordName);
	m_pooldb->commit();
	m_pooldb->disconnect();	
	cond::MetaData metadata(*m_coraldb);
	m_coraldb->connect(cond::ReadWriteCreate);
	m_coraldb->startTransaction(false);
	metadata.addMapping(myrecord.m_tag,iovToken);
	m_coraldb->commit();
	m_coraldb->disconnect();
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
	m_pooldb->connect();
	m_pooldb->startTransaction(false);    
	cond::Ref<T> myPayload(*m_pooldb,payloadObj);
	myPayload.markWrite(EventSetupRecordName);
	std::string payloadToken=myPayload.token();
	std::string iovToken=this->insertIOV(myrecord,payloadToken,tillTime,EventSetupRecordName);
	m_pooldb->commit();    
	m_pooldb->disconnect();
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
	m_pooldb->connect();
	m_pooldb->startTransaction(false);    
	cond::Ref<T> myPayload(*m_pooldb,payloadObj);
	myPayload.markWrite(EventSetupRecordName);
	std::string payloadToken=myPayload.token();
	this->appendIOV(myrecord,payloadToken,sinceTime);
	m_pooldb->commit();
	m_pooldb->disconnect();
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
      //std::string m_connect;
      //std::string m_timetype;
      //std::string m_catalog;
      //cond::Time_t m_endOfTime;
      cond::Time_t m_currentTime;
      cond::DBSession* m_session;
      cond::IOVService* m_iovservice;
      cond::PoolStorageManager* m_pooldb;
      cond::RelationalStorageManager* m_coraldb;
      std::map<size_t, cond::service::serviceCallbackRecord> m_callbacks;
      bool m_dbstarted;
      //edm::ParameterSet m_connectionPset;
    };//PoolDBOutputService
  }//ns service
}//ns cond
#endif
