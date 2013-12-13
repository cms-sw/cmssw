#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/Exception.h"
//#include "CondCore/DBCommon/interface/TagInfo.h"
//#include "CondCore/DBCommon/interface/IOVInfo.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/CondDB/interface/Exception.h"
//
#include <vector>
#include<memory>

void 
cond::service::PoolDBOutputService::fillRecord( edm::ParameterSet & pset) {
  Record thisrecord;

  thisrecord.m_idName = pset.getParameter<std::string>("record");
  thisrecord.m_tag = pset.getParameter<std::string>("tag");
  
  thisrecord.m_closeIOV =
    pset.getUntrackedParameter<bool>("closeIOV", m_closeIOV);
 
  //thisrecord.m_timetype=cond::findSpecs(pset.getUntrackedParameter< std::string >("timetype",m_timetypestr)).type;
  thisrecord.m_timetype = cond::time::timeTypeFromName( pset.getUntrackedParameter< std::string >("timetype",m_timetypestr) );

  m_callbacks.insert(std::make_pair(thisrecord.m_idName,thisrecord));

  // *** THE LOGGING has still to be defined and implemented.
  //if( !m_logConnectionString.empty() ){
  //    cond::UserLogInfo userloginfo;
  //    m_logheaders.insert(std::make_pair(thisrecord.m_idName,userloginfo));
  //}
}

cond::service::PoolDBOutputService::PoolDBOutputService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & iAR ): 
  m_timetypestr(""),
  m_currentTime( 0 ),
  m_session(),
  //m_logConnectionString(""),
  //m_logdb(),
  m_dbstarted( false ),
  m_callbacks(),
  //m_newtags(),
  m_closeIOV(false)//,
  //m_logheaders()
{
  m_closeIOV=iConfig.getUntrackedParameter<bool>("closeIOV",m_closeIOV);

  m_timetypestr=iConfig.getUntrackedParameter< std::string >("timetype","runnumber");
  m_timetype = cond::time::timeTypeFromName( m_timetypestr );
  
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters");
  cond::persistency::ConnectionPool connection;
  connection.setParameters( connectionPset );
  connection.configure();

  m_session = connection.createSession( iConfig.getParameter<std::string>("connect"), true ); 
  
  //if( iConfig.exists("logconnect") ){
  //  m_logConnectionString = iConfig.getUntrackedParameter<std::string>("logconnect");
  //  cond::DbSession logSession = connection.createSession();
  //  m_logdb.reset( new cond::Logger( logSession ) );
  //}  
  
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toPut=iConfig.getParameter<Parameters>("toPut");
  for(Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut)
    fillRecord( *itToPut);


  iAR.watchPreProcessEvent(this,&cond::service::PoolDBOutputService::preEventProcessing);
  iAR.watchPostEndJob(this,&cond::service::PoolDBOutputService::postEndJob);
  iAR.watchPreModule(this,&cond::service::PoolDBOutputService::preModule);
  iAR.watchPostModule(this,&cond::service::PoolDBOutputService::postModule);
  iAR.watchPreBeginLumi(this,&cond::service::PoolDBOutputService::preBeginLumi);
}

cond::persistency::Session
cond::service::PoolDBOutputService::session() const{
  return m_session;
}

std::string 
cond::service::PoolDBOutputService::tag( const std::string& recordName ){
  return this->lookUpRecord(recordName).m_tag;
}

bool 
cond::service::PoolDBOutputService::isNewTagRequest( const std::string& recordName ){
  Record& myrecord=this->lookUpRecord(recordName);
  return myrecord.m_isNewTag; 
}

void 
cond::service::PoolDBOutputService::initDB( bool forReading )
{
  m_session.transaction().start(false);
  cond::persistency::TransactionScope scope( m_session.transaction() );
  try{ 
    if(!forReading) {
      if( !m_session.existsDatabase() ) m_session.createDatabase();
    }
    //init logdb if required
    //if(!m_logConnectionString.empty()){
    //  m_logdb->connect( m_logConnectionString );
    //  m_logdb->createLogDBIfNonExist();
    //}
  } catch( const std::exception& er ){
    cond::throwException( std::string(er.what()),"PoolDBOutputService::initDB" );
  }
  scope.close();
  m_dbstarted=true;
}

void 
cond::service::PoolDBOutputService::postEndJob()
{
  if( m_dbstarted) {
    m_session.transaction().commit();
    m_dbstarted = false;
  }
}

void 
cond::service::PoolDBOutputService::preEventProcessing(const edm::EventID& iEvtid, const edm::Timestamp& iTime)
{
  if( m_timetype == cond::runnumber ){//runnumber
    m_currentTime=iEvtid.run();
  }else if( m_timetype == cond::timestamp ){ //timestamp
    m_currentTime=iTime.value();
  }
}

void
cond::service::PoolDBOutputService::preModule(const edm::ModuleDescription& desc){
}

void 
cond::service::PoolDBOutputService::preBeginLumi(const edm::LuminosityBlockID& iLumiid,  const edm::Timestamp& iTime ){
  if( m_timetype == cond::lumiid ){
    m_currentTime=iLumiid.value();
  }
}

void
cond::service::PoolDBOutputService::postModule(const edm::ModuleDescription& desc){
}

cond::service::PoolDBOutputService::~PoolDBOutputService(){
}


cond::Time_t 
cond::service::PoolDBOutputService::endOfTime() const{
  return timeTypeSpecs[m_timetype].endValue;
}

cond::Time_t 
cond::service::PoolDBOutputService::beginOfTime() const{
  return timeTypeSpecs[m_timetype].beginValue;
}

cond::Time_t 
cond::service::PoolDBOutputService::currentTime() const{
  return m_currentTime;
}

void 
cond::service::PoolDBOutputService::createNewIOV( const std::string& firstPayloadId,
						  const std::string payloadType, 
                                                  cond::Time_t firstSinceTime, 
                                                  cond::Time_t firstTillTime,
                                                  const std::string& recordName, 
                                                  bool withlogging){
  cond::persistency::TransactionScope scope( m_session.transaction() );
  Record& myrecord=this->lookUpRecord(recordName);
  if(!myrecord.m_isNewTag) {
    cond::throwException( myrecord.m_tag + " is not a new tag", "PoolDBOutputService::createNewIOV");
  }
  std::string iovToken;
  //if(withlogging){
  //  if( m_logConnectionString.empty() ) {
  //    throw cond::db::Exception("Log db was not set from PoolDBOutputService::createNewIOV",
  //				"PoolDBOutputService::createNewIOV");
  //  }
  //}
 
  try{
    // FIX ME: synchronization type and description have to be passed as the other parameters?
    cond::persistency::IOVEditor editor = m_session.createIov( payloadType, myrecord.m_tag, myrecord.m_timetype, cond::OFFLINE ); 
    editor.setDescription( "New Tag" );
    editor.insert( firstSinceTime, firstPayloadId );
    editor.flush();
    myrecord.m_isNewTag=false;
    //if(withlogging){
    //  std::string destconnect=m_session.connectionString();
    //  cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
    //  m_logdb->logOperationNow(a,destconnect,objClass,objToken,myrecord.m_tag,myrecord.timetypestr(),payloadIdx,firstSinceTime);
    //}
  }catch(const std::exception& er){ 
    //if(withlogging){
    //  std::string destconnect=m_session.connectionString();
    //  cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
    //  m_logdb->logFailedOperationNow(a,destconnect,objClass,objToken,myrecord.m_tag,myrecord.timetypestr(),payloadIdx,firstSinceTime,std::string(er.what()));
    //}
    cond::throwException(std::string(er.what()) + " from PoolDBOutputService::createNewIOV ",
		   "PoolDBOutputService::createNewIOV");
  }
  scope.close();
}

void 
cond::service::PoolDBOutputService::createNewIOV( const std::string& firstPayloadId,
                                                  cond::Time_t firstSinceTime, 
                                                  cond::Time_t firstTillTime,
                                                  const std::string& recordName, 
                                                  bool withlogging){
  cond::persistency::TransactionScope scope( m_session.transaction() );
  Record& myrecord=this->lookUpRecord(recordName);
  if(!myrecord.m_isNewTag) {
    cond::throwException( myrecord.m_tag + " is not a new tag", "PoolDBOutputService::createNewIOV");
  }
  std::string iovToken;
  try{
    // FIX ME: synchronization type and description have to be passed as the other parameters?
    cond::persistency::IOVEditor editor = m_session.createIovForPayload( firstPayloadId, myrecord.m_tag, myrecord.m_timetype, cond::OFFLINE ); 
    editor.setDescription( "New Tag" );
    editor.insert( firstSinceTime, firstPayloadId );
    editor.flush();
    myrecord.m_isNewTag=false;
  }catch(const std::exception& er){ 
    cond::throwException(std::string(er.what()) + " from PoolDBOutputService::createNewIOV ",
		   "PoolDBOutputService::createNewIOV");
  }
  scope.close();
}

void 
cond::service::PoolDBOutputService::appendSinceTime( const std::string& payloadId,
						     cond::Time_t time,
						     const std::string& recordName,
						     bool withlogging) {
  cond::persistency::TransactionScope scope( m_session.transaction() );
  Record& myrecord=this->lookUpRecord(recordName);
  if( myrecord.m_isNewTag ) {
    cond::throwException(std::string("Cannot append to non-existing tag ") + myrecord.m_tag,
		   "PoolDBOutputService::appendSinceTime");  
  }
  //if(withlogging){
  //  if( m_logConnectionString.empty() ) {
  //     throw cond::Exception("Log db was not set from PoolDBOutputService::add");
  //  }
  //}

  try{
    cond::persistency::IOVEditor editor = m_session.editIov( myrecord.m_tag ); 
    editor.insert( time, payloadId );
    editor.flush();

    //if(withlogging){
    //  std::string destconnect=m_session.connectionString();
    //  cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
    //  m_logdb->logOperationNow(a,destconnect,objClass,objToken,myrecord.m_tag,myrecord.timetypestr(),payloadIdx,time);
    //}
  }catch(const std::exception& er){
    //if(withlogging){
    //  std::string destconnect=m_session.connectionString();
    //  cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
    //  m_logdb->logFailedOperationNow(a,destconnect,objClass,objToken,myrecord.m_tag,myrecord.timetypestr(),payloadIdx,time,std::string(er.what()));
    //}
    cond::throwException(std::string(er.what()),
		   "PoolDBOutputService::appendSinceTime");
  }
  scope.close();
}

cond::service::PoolDBOutputService::Record& 
cond::service::PoolDBOutputService::lookUpRecord(const std::string& recordName){
  if (!m_dbstarted) this->initDB( false );
  cond::persistency::TransactionScope scope( m_session.transaction() );
  std::map<std::string,Record>::iterator it=m_callbacks.find(recordName);
  if(it==m_callbacks.end()) {
    cond::throwException("The record \""+recordName +"\" has not been registered.","PoolDBOutputService::lookUpRecord");
  }
  if( !m_session.existsIov( it->second.m_tag) ){
    it->second.m_isNewTag=true;
  } else {
    it->second.m_isNewTag=false;
  }
  scope.close();
  return it->second;
}

//cond::UserLogInfo& 
//cond::service::PoolDBOutputService::lookUpUserLogInfo(const std::string& recordName){
//  std::map<std::string,cond::UserLogInfo>::iterator it=m_logheaders.find(recordName);
//  if(it==m_logheaders.end()) throw cond::Exception("Log db was not set for record " + recordName + " from PoolDBOutputService::lookUpUserLogInfo");
//  return it->second;
//}

void 
cond::service::PoolDBOutputService::closeIOV(Time_t lastTill, const std::string& recordName, 
					     bool withlogging) {
  // not fully working.. not be used for now...
  Record & myrecord  = lookUpRecord(recordName);
  cond::persistency::TransactionScope scope( m_session.transaction() );

  if( myrecord.m_isNewTag ) {
    cond::throwException(std::string("Cannot close non-existing tag ") + myrecord.m_tag,
			 "PoolDBOutputService::closeIOV");
  }
  cond::persistency::IOVEditor editor = m_session.editIov( myrecord.m_tag ); 
  editor.setEndOfValidity( lastTill );
  editor.flush();
  scope.close();
}


void
cond::service::PoolDBOutputService::setLogHeaderForRecord(const std::string& recordName,const std::string& dataprovenance,const std::string& usertext)
{
  //cond::UserLogInfo& myloginfo=this->lookUpUserLogInfo(recordName);
  //myloginfo.provenance=dataprovenance;
  //myloginfo.usertext=usertext;
}

//
//const cond::Logger& 
//cond::service::PoolDBOutputService::queryLog()const{
//  if( !m_logdb.get() ) throw cond::Exception("Log database is not set from PoolDBOutputService::queryLog");
//  return *m_logdb;
//}

// Still required.
void 
cond::service::PoolDBOutputService::tagInfo(const std::string& recordName,cond::TagInfo_t& result ){
  //
  Record& record = lookUpRecord(recordName);
  result.name=record.m_tag;
  //use iovproxy to find out.
  cond::persistency::IOVProxy iov = m_session.readIov( record.m_tag );
  result.size=iov.sequenceSize();
  if (result.size>0) {
    cond::Iov_t last = iov.getLast();
    result.lastInterval = cond::ValidityInterval( last.since, last.till );
    result.lastPayloadToken = last.payloadId;
  }
}
