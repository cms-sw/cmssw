#ifndef CondCore_PoolDBOutputService_h
#define CondCore_PoolDBOutputService_h
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "serviceCallbackRecord.h"
#include "Logger.h"
#include "UserLogInfo.h"
#include "TagInfo.h"
#include <string>
#include <map>
//#include <iostream>
//
// Package:     DBOutputService
// Class  :     PoolDBOutputService
// 
/**\class PoolDBOutputService PoolDBOutputService.h CondCore/DBOutputService/interface/PoolDBOutputService.h
   Description: edm service for writing conditions object to DB.  
*/
//
// Author:      Zhen Xie
//
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


    struct GetToken {
      virtual std::string operator()(cond::PoolTransaction&) const =0;

    };

    struct GetTrivialToken : public GetToken {
      
      GetTrivialToken(std::string token) : 
	m_token(token){}
      virtual ~GetTrivialToken(){}
      virtual std::string operator()(cond::PoolTransaction&) const {
	return m_token;
      }

      std::string m_token;
    };

    template<typename T>
    struct GetTokenFromPointer : public GetToken {
      
      GetTokenFromPointer(T * p, const std::string& recordName) : 
	m_p(p),  m_recordName(recordName) {}
      
      virtual std::string operator()(cond::PoolTransaction& pooldb) const {
	cond::TypedRef<T> myPayload(pooldb,m_p);
	  myPayload.markWrite(m_recordName);
	  return myPayload.token();

      }

      T* m_p;
      const std::string& m_recordName;
    };



      
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
      //
      // return the database connection handle in use
      //
      cond::Connection& connection() const;
      // 
      std::string tag( const std::string& EventSetupRecordName );
      bool isNewTagRequest( const std::string& EventSetupRecordName );
      const cond::Logger& queryLog() const;


      // 180 compatible interface
      template<typename T>
      void createNewIOV( T* firstPayloadObj,
			 cond::Time_t firstTillTime,
			 const std::string& EventSetupRecordName){
	// generate warning
	//bool UsingTheOldInterfaceWOfirstSinceTimePleaseUpgrade;
	
	createNewIOV(firstPayloadObj, beginOfTime(),firstTillTime, EventSetupRecordName,false);
      }

      //
      // insert the payload and its valid till time into the database
      // Note: user looses the ownership of the pointer to the payloadObj
      // The payload object will be stored as well
      // 
      template<typename T>
	void createNewIOV( T* firstPayloadObj, 
			   cond::Time_t firstSinceTime,
			   cond::Time_t firstTillTime,
			   const std::string& EventSetupRecordName,
			   bool withlogging=false){

	createNewIOV( GetTokenFromPointer<T>(firstPayloadObj,EventSetupRecordName),
		      firstSinceTime, 
		      firstTillTime,
		      EventSetupRecordName,
		      withlogging);
	
      }

      void createNewIOV( const std::string& firstPayloadToken, 
			 cond::Time_t firstSinceTime, 
			 cond::Time_t firstTillTime,
			 const std::string& EventSetupRecordName,
			 bool withlogging=false) {
	
	createNewIOV( GetTrivialToken(firstPayloadToken),
		      firstSinceTime, 
		      firstTillTime,
		      EventSetupRecordName,
		      withlogging);
      }

      template<typename T>
      void appendTillTime( T* payloadObj, 
			   cond::Time_t tillTime,
			   const std::string& EventSetupRecordName,
			   bool withlogging=false){
	add(false,
	    GetTokenFromPointer<T>(payloadObj,EventSetupRecordName),
	    tillTime, 
	    EventSetupRecordName,
	    withlogging);
      }

      void appendTillTime( const std::string& payloadToken, 
			   cond::Time_t tillTime,
			   const std::string& EventSetupRecordName,
			   bool withlogging=false
			   ) {

	add(false,
	    GetTrivialToken(payloadToken),
	    tillTime, 
	    EventSetupRecordName,
	    withlogging);
      }

      
      template<typename T>
	void appendSinceTime( T* payloadObj, 
			      cond::Time_t sinceTime,
			      const std::string& EventSetupRecordName,
			      bool withlogging=false){
	add(true,
	    GetTokenFromPointer<T>(payloadObj,EventSetupRecordName),
	    sinceTime, 
	    EventSetupRecordName,
	    withlogging);
      }

      // Append the payload and its valid sinceTime into the database
      // Note: user looses the ownership of the pointer to the payloadObj
      // Note: the iov index appended to MUST pre-existing and the existing 
      // conditions data are retrieved from EventSetup 
      // 
      void appendSinceTime( const std::string& payloadToken, 
			   cond::Time_t sinceTime,
			   const std::string& EventSetupRecordName,
			    bool withlogging=false) {
	add(true,
	    GetTrivialToken(payloadToken),
	    sinceTime, 
	    EventSetupRecordName,
	    withlogging);	
      }


      // write one (either create or append
      template<typename T>
      void writeOne(T * payload, Time_t time, const std::string& recordName, 
		    bool withlogging=false, bool since=true) {
	if (isNewTagRequest(recordName) ){
	  createNewIOV<T>(payload, 
			  since ? time : beginOfTime(),
			  since ?  endOfTime() : time, 
			  recordName, withlogging);
	}
	else{
	  if (since){ 
	    appendSinceTime<T>(payload, time, recordName, withlogging);
	  } 
	  else { 
	    appendTillTime<T>(payload, time, recordName, withlogging);
	  }
	}	
      }


      //
      // Service time utility callback method 
      // return the infinity value according to the given timetype
      //
      cond::Time_t endOfTime() const;
      //
      // Service time utility callback method 
      // return beginning of time value according to the given timetype
      //
      cond::Time_t beginOfTime() const;
      //
      // Service time utility callback method 
      // return the current conditions time value according to the 
      // given timetype
      //
      cond::Time_t currentTime() const;
      // optional. User can inject additional information into the log associated with a given record
      void setLogHeaderForRecord(const std::string& EventSetupRecordName,
			   const std::string& provenance,
			   const std::string& usertext);
      // 
      // Retrieve tag information of the data
      // 
      void tagInfo(const std::string& EventSetupRecordName,
		   cond::TagInfo& result );
      virtual ~PoolDBOutputService();  

    private:

      void createNewIOV( GetToken const & token, 
			 cond::Time_t firstSinceTime, 
			 cond::Time_t firstTillTime,
			 const std::string& EventSetupRecordName,
			 bool withlogging=false);

      void add( bool sinceNotTill, 
		  GetToken const & token,  
		   cond::Time_t time,
		   const std::string& EventSetupRecordName,
		   bool withlogging=false);


      void connect();    
      void disconnect();
      void initDB();
      size_t callbackToken(const std::string& EventSetupRecordName ) const ;
      unsigned int appendIOV(cond::PoolTransaction&,
			     cond::service::serviceCallbackRecord& record,
			     const std::string& payloadToken, 
			     cond::Time_t sinceTime);

      /// Returns payload location index 
      unsigned int 
      insertIOV(cond::PoolTransaction& pooldb,
		cond::service::serviceCallbackRecord& record,
		const std::string& payloadToken, 			    
		cond::Time_t tillTime);
      //			    const std::string& EventSetupRecordName);
      
      serviceCallbackRecord& lookUpRecord(const std::string& EventSetupRecordName);
      UserLogInfo& lookUpUserLogInfo(const std::string& EventSetupRecordName);
      
    private:
      cond::TimeType m_timetype; 
      std::string m_timetypestr;
      cond::Time_t m_currentTime;
      cond::DBSession* m_session;
      cond::Connection* m_connection;
      std::map<size_t, cond::service::serviceCallbackRecord> m_callbacks;
      std::vector< std::pair<std::string,std::string> > m_newtags;
      bool m_dbstarted;
      cond::Logger* m_logdb;
      bool m_logdbOn;
      std::map<size_t, cond::service::UserLogInfo> m_logheaders;
      //cond::IOVService* m_iovservice;
      //edm::ParameterSet m_connectionPset;
    };//PoolDBOutputService
  }//ns service
}//ns cond
#endif
