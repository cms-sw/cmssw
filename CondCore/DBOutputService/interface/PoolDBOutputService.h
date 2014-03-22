#ifndef CondCore_PoolDBOutputService_h
#define CondCore_PoolDBOutputService_h
#include "FWCore/ServiceRegistry/interface/ActivityRegistry.h"
#include "FWCore/Utilities/interface/TypeID.h"
//#include "CondCore/DBCommon/interface/Logger.h"
//#include "CondCore/DBCommon/interface/LogDBEntry.h"
//#include "CondCore/DBCommon/interface/TagInfo.h"
#include "CondCore/CondDB/interface/Session.h"
#include <string>
#include <map>

//
// Package:     DBOutputService
// Class  :     PoolDBOutputService
// 
/**\class PoolDBOutputService PoolDBOutputService.h CondCore/DBOutputService/interface/PoolDBOutputService.h
   Description: edm service for writing conditions object to DB.  
*/
//
// Author:      Zhen Xie
// Fixes and other changes: Giacomo Govi
//

namespace edm{
  class Event;
  class EventSetup;
  class ParameterSet;
}
namespace cond{
  
  inline std::string classNameForTypeId( const std::type_info& typeInfo ){
    edm::TypeID type( typeInfo );
    return type.className();
  }
  
  namespace service {

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
      // return the database session in use ( GG: not sure this is still useful... )
      //
      cond::persistency::Session session() const;
      //
      std::string tag( const std::string& recordName );
      bool isNewTagRequest( const std::string& recordName );
      //const cond::Logger& queryLog() const;
      
      // 
      template<typename T>
      void writeOne( T * payload, Time_t time, const std::string& recordName, bool withlogging=false ) {
        if( !payload ) throwException( "Provided payload pointer is invalid.","PoolDBOutputService::writeOne");
	if (!m_dbstarted) this->initDB( false );
	Hash payloadId = m_session.storePayload( *payload );
	std::string payloadType = cond::demangledName(typeid(T));
	if (isNewTagRequest(recordName) ){
	  createNewIOV(payloadId, payloadType, time, endOfTime(), recordName, withlogging);
        } else {
	  appendSinceTime(payloadId, time, recordName, withlogging);
        }	
      }

      // close the IOVSequence setting lastTill
      void closeIOV(Time_t lastTill, const std::string& recordName, 
                    bool withlogging=false);

      // 
      template<typename T>
      void createNewIOV( T* firstPayloadObj,
			 cond::Time_t firstSinceTime,
			 cond::Time_t firstTillTime,
			 const std::string& recordName,
                         bool withlogging=false){
        if( !firstPayloadObj ) throwException( "Provided payload pointer is invalid.","PoolDBOutputService::createNewIOV");
	if (!m_dbstarted) this->initDB( false );
        createNewIOV( m_session.storePayload( *firstPayloadObj ),
		      cond::demangledName(typeid(T)),
                      firstSinceTime,
                      firstTillTime,
                      recordName,
                      withlogging);	
      }
            
      void createNewIOV( const std::string& firstPayloadId,
			 const std::string payloadType, 
                         cond::Time_t firstSinceTime,
                         cond::Time_t firstTillTime,
                         const std::string& recordName,
                         bool withlogging=false);
      
      // this one we need to avoid to adapt client code around... to be removed in the long term!
      void createNewIOV( const std::string& firstPayloadId,
                         cond::Time_t firstSinceTime,
                         cond::Time_t firstTillTime,
                         const std::string& recordName,
                         bool withlogging=false);

      // 
      template<typename T> void appendSinceTime( T* payloadObj,
                                                 cond::Time_t sinceTime,
                                                 const std::string& recordName,
                                                 bool withlogging=false){
        if( !payloadObj ) throwException( "Provided payload pointer is invalid.","PoolDBOutputService::appendSinceTime");
        appendSinceTime( m_session.storePayload( *payloadObj ),
			 sinceTime,
			 recordName,
			 withlogging);
      }
      
      // Append the payload and its valid sinceTime into the database
      // Note: the iov index appended to MUST pre-existing and the existing 
      // conditions data are retrieved from the DB
      // 
      void appendSinceTime( const std::string& payloadId,
                            cond::Time_t sinceTime,
                            const std::string& recordName,
                            bool withlogging=false);
     
      //
      // Service time utility method 
      // return the infinity value according to the given timetype
      //
      cond::Time_t endOfTime() const;
      //
      // Service time utility method 
      // return beginning of time value according to the given timetype
      //
      cond::Time_t beginOfTime() const;
      //
      // Service time utility method 
      // return the time value of the current edm::Event according to the 
      // given timetype
      //
      cond::Time_t currentTime() const;

      // optional. User can inject additional information into the log associated with a given record
      void setLogHeaderForRecord(const std::string& recordName,
				 const std::string& provenance,
				 const std::string& usertext);
      // 
      // Retrieve tag information of the data
      // 
      void tagInfo(const std::string& recordName,
      		   cond::TagInfo_t& result );
      
      virtual ~PoolDBOutputService();  
      
    private:

      struct Record{
	Record(): m_tag(),
		  m_isNewTag(false),
		  m_idName(),
		  m_timetype(cond::runnumber),
                  m_closeIOV(false)
	{}

	std::string timetypestr() const { return cond::timeTypeSpecs[m_timetype].name;}
	std::string m_tag;
	bool m_isNewTag;
	std::string m_idName;
	cond::TimeType m_timetype;
        bool m_closeIOV;
    };      



      void fillRecord( edm::ParameterSet & pset);
      
      void connect();    
      void disconnect();
      void initDB( bool forReading=true );

      Record & lookUpRecord(const std::string& recordName);
      //cond::UserLogInfo& lookUpUserLogInfo(const std::string& recordName);
      
    private:
      cond::TimeType m_timetype; 
      std::string m_timetypestr;
      cond::Time_t m_currentTime;

      cond::persistency::Session m_session;
      //std::string m_logConnectionString;
      //std::auto_ptr<cond::Logger> m_logdb;
      bool m_dbstarted;

      std::map<std::string, Record> m_callbacks;
      //std::vector< std::pair<std::string,std::string> > m_newtags;
      bool m_closeIOV;
      //std::map<std::string, cond::UserLogInfo> m_logheaders;

    };//PoolDBOutputService
  }//ns service
}//ns cond
#endif
