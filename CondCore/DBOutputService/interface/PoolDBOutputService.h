#ifndef CondCore_PoolDBOutputService_h
#define CondCore_PoolDBOutputService_h
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "CondCore/DBCommon/interface/PoolTransaction.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
#include "CondCore/DBCommon/interface/TypedRef.h"
#include "CondCore/DBCommon/interface/Time.h"
#include "CondCore/MetaDataService/interface/MetaData.h"
#include "serviceCallbackRecord.h"
#include "CondCore/DBCommon/interface/Logger.h"
#include "CondCore/DBCommon/interface/UserLogInfo.h"
#include "CondCore/DBCommon/interface/TagInfo.h"
#include <string>
#include <map>
#include "CondFormats/Common/interface/PayloadWrapper.h"

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
      virtual std::string operator()(cond::PoolTransaction&, bool) const =0;
      static unsigned int sizeDSW();
    };

    struct GetTrivialToken : public GetToken {
      
      GetTrivialToken(std::string token) : 
	m_token(token){}
      virtual ~GetTrivialToken(){}
      virtual std::string operator()(cond::PoolTransaction&, bool) const {
	return m_token;
      }

      std::string m_token;
    };

    template<typename T>
    struct GetTokenFromPointer : public GetToken {
      typedef cond::DataWrapper<T> Wrapper;

      GetTokenFromPointer(T * p, Summary * s) : 
	m_p(p), m_s(s){}
	//m_w(new Wrapper(p,s)){}
      
      virtual std::string operator()(cond::PoolTransaction& pooldb, bool=withWrapper) const {
	if (withWrapper) {
	  cond::TypedRef<Wrapper> myPayload(pooldb,new Wrapper(p,s));
	  myPayload.markWrite(myPayload.className().replace(0,sizeDSW(),"DSW"));
	  return myPayload.token();
	} else {
	  cond::TypedRef<T> myPayload(pooldb,p);
	  myPayload.markWrite(myPayload.className());
	  return myPayload.token();
	}
	return "make compiler happy";
      }

      T * m_p;
      S * m_s;
      // Wrapper * m_w;
      //const std::string& m_recordName;
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
      void preBeginLumi(const edm::LuminosityBlockID&, 
			const edm::Timestamp& );
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


      /* write one (either create or append)
       * The ONE and ONLY interface supportd in future!
       */
      template<typename T>
      void writeOne(T * payload, Summary * summary, 
		    Time_t time, const std::string& recordName, 
		    bool withlogging=false, bool since=true) {
	if (isNewTagRequest(recordName) ){
	  createNewIOV<T>(payload, summary,
			  since ? time : beginOfTime(),
			  since ?  endOfTime() : time, 
			  recordName, withlogging);
	}
	else{
	  if (since){ 
	    appendSinceTime<T>(payload, summary, time, recordName, withlogging);
	  } 
	  else { 
	    appendTillTime<T>(payload, summary, time, recordName, withlogging);
	  }
	}	
      }




      //
      // insert the payload and its valid since/till time into the database
      // Note: user looses the ownership of the pointer to the payloadObj
      // The payload object will be stored as well
      // 
      template<typename T>
	void createNewIOV( T* firstPayloadObj,  Summary * summary,
			   cond::Time_t firstSinceTime,
			   cond::Time_t firstTillTime,
			   const std::string& EventSetupRecordName,
			   bool withlogging=false){

	createNewIOV( GetTokenFromPointer<T>(firstPayloadObj, summary),
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
      void appendTillTime( T* payloadObj,  Summary * summary,
			   cond::Time_t tillTime,
			   const std::string& EventSetupRecordName,
			   bool withlogging=false){
	add(false,
	    GetTokenFromPointer<T>(payloadObj,summary),
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
      void appendSinceTime( T* payloadObj, Summary * summary,
			      cond::Time_t sinceTime,
			      const std::string& EventSetupRecordName,
			      bool withlogging=false){
	add(true,
	    GetTokenFromPointer<T>(payloadObj,summary),
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
      cond::UserLogInfo& lookUpUserLogInfo(const std::string& EventSetupRecordName);
      
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
      std::map<size_t, cond::UserLogInfo> m_logheaders;
      //cond::IOVService* m_iovservice;
      //edm::ParameterSet m_connectionPset;
    };//PoolDBOutputService
  }//ns service
}//ns cond
#endif
