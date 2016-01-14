#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/Exception.h"
#include "CondCore/CondDB/interface/ConnectionPool.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ServiceRegistry/interface/StreamContext.h"
#include "FWCore/ServiceRegistry/interface/GlobalContext.h"
#include "FWCore/ServiceRegistry/interface/SystemBounds.h"
#include "CondCore/CondDB/interface/Exception.h"
//
#include <vector>
#include<memory>
#include <cassert>

//In order to make PoolDBOutputService::currentTime() to work we have to keep track
// of which stream is presently being processed on a given thread during the call of
// a module which calls that method.
static thread_local int s_streamIndex = -1;

void 
cond::service::PoolDBOutputService::fillRecord( edm::ParameterSet & pset) {
  Record thisrecord;

  thisrecord.m_idName = pset.getParameter<std::string>("record");
  thisrecord.m_tag = pset.getParameter<std::string>("tag");
  
  thisrecord.m_closeIOV =
    pset.getUntrackedParameter<bool>("closeIOV", m_closeIOV);
 
  thisrecord.m_timetype = cond::time::timeTypeFromName( pset.getUntrackedParameter< std::string >("timetype",m_timetypestr) );

  m_callbacks.insert(std::make_pair(thisrecord.m_idName,thisrecord));

  cond::UserLogInfo userloginfo;
  m_logheaders.insert(std::make_pair(thisrecord.m_idName,userloginfo));
}

cond::service::PoolDBOutputService::PoolDBOutputService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & iAR ): 
  m_timetypestr(""),
  m_currentTimes{},
  m_session(),
  m_dbstarted( false ),
  m_callbacks(),
  m_closeIOV(false),
  m_logheaders()
{
  m_closeIOV=iConfig.getUntrackedParameter<bool>("closeIOV",m_closeIOV);

  m_timetypestr=iConfig.getUntrackedParameter< std::string >("timetype","runnumber");
  m_timetype = cond::time::timeTypeFromName( m_timetypestr );
  
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters");
  cond::persistency::ConnectionPool connection;
  connection.setParameters( connectionPset );
  connection.configure();
  std::string connectionString = iConfig.getParameter<std::string>("connect");
  m_session = connection.createSession( connectionString, true ); 
  
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toPut=iConfig.getParameter<Parameters>("toPut");
  for(Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut)
    fillRecord( *itToPut);


  iAR.watchPostEndJob(this,&cond::service::PoolDBOutputService::postEndJob);
  iAR.watchPreallocate([this](edm::service::SystemBounds const& iBounds) {
      m_currentTimes.resize(iBounds.maxNumberOfStreams());
    });
  if( m_timetype == cond::timestamp ){ //timestamp
    iAR.watchPreEvent(this,&cond::service::PoolDBOutputService::preEventProcessing);
    iAR.watchPreModuleEvent(this, &cond::service::PoolDBOutputService::preModuleEvent);
    iAR.watchPostModuleEvent(this, &cond::service::PoolDBOutputService::postModuleEvent);
  } else if( m_timetype == cond::runnumber ){//runnumber
    //NOTE: this assumes only one run is being processed at a time.
    // This is true for 7_1_X but plan are to allow multiple in flight at a time
    s_streamIndex = 0;
    iAR.watchPreGlobalBeginRun(this,&cond::service::PoolDBOutputService::preGlobalBeginRun);
  } else if( m_timetype == cond::lumiid ){
    //NOTE: this assumes only one lumi is being processed at a time.
    // This is true for 7_1_X but plan are to allow multiple in flight at a time
    s_streamIndex = 0;
    iAR.watchPreGlobalBeginLumi(this,&cond::service::PoolDBOutputService::preGlobalBeginLumi);
  }
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
cond::service::PoolDBOutputService::initDB( bool )
{
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  cond::persistency::TransactionScope scope( m_session.transaction() );
  scope.start( false );
  try{ 
    if( !m_session.existsDatabase() ) m_session.createDatabase();

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
cond::service::PoolDBOutputService::preEventProcessing(edm::StreamContext const& iContext)
{
  m_currentTimes[iContext.streamID().value()] = iContext.timestamp().value();
}

void 
cond::service::PoolDBOutputService::preModuleEvent(edm::StreamContext const& iContext, edm::ModuleCallingContext const&) {
  s_streamIndex = iContext.streamID().value();
}

void 
cond::service::PoolDBOutputService::postModuleEvent(edm::StreamContext const& iContext, edm::ModuleCallingContext const&) {
  s_streamIndex = -1;
}

void 
cond::service::PoolDBOutputService::preGlobalBeginRun(edm::GlobalContext const& iContext) {
  for( auto& time : m_currentTimes) {
    time = iContext.luminosityBlockID().run();
  }
}

void 
cond::service::PoolDBOutputService::preGlobalBeginLumi(edm::GlobalContext const& iContext) {
  for( auto& time : m_currentTimes) {
    time = iContext.luminosityBlockID().value();
  }
}

cond::service::PoolDBOutputService::~PoolDBOutputService(){
  if( m_dbstarted) {
    m_session.transaction().rollback();
  }
}

void cond::service::PoolDBOutputService::forceInit(){
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
    if (!m_dbstarted) initDB();  
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
  assert(-1 != s_streamIndex);
  return m_currentTimes[s_streamIndex];
}

void 
cond::service::PoolDBOutputService::createNewIOV( const std::string& firstPayloadId,
						  const std::string payloadType, 
                                                  cond::Time_t firstSinceTime, 
                                                  cond::Time_t firstTillTime,
                                                  const std::string& recordName, 
                                                  bool withlogging){
  std::lock_guard<std::recursive_mutex> lock(m_mutex);

  cond::persistency::TransactionScope scope( m_session.transaction() );
  Record& myrecord=this->lookUpRecord(recordName);
  if(!myrecord.m_isNewTag) {
    cond::throwException( myrecord.m_tag + " is not a new tag", "PoolDBOutputService::createNewIOV");
  }
  std::string iovToken;

  try{
    // FIX ME: synchronization type and description have to be passed as the other parameters?
    cond::persistency::IOVEditor editor = m_session.createIov( payloadType, myrecord.m_tag, myrecord.m_timetype, cond::SYNCH_ANY ); 
    editor.setDescription( "New Tag" );
    editor.insert( firstSinceTime, firstPayloadId );
    cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
    editor.flush( a.usertext );
    myrecord.m_isNewTag=false;
  }catch(const std::exception& er){ 
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
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  cond::persistency::TransactionScope scope( m_session.transaction() );
  Record& myrecord=this->lookUpRecord(recordName);
  if(!myrecord.m_isNewTag) {
    cond::throwException( myrecord.m_tag + " is not a new tag", "PoolDBOutputService::createNewIOV");
  }
  std::string iovToken;
  std::string payloadType("");
  try{
    // FIX ME: synchronization type and description have to be passed as the other parameters?
    cond::persistency::IOVEditor editor = m_session.createIovForPayload( firstPayloadId, myrecord.m_tag, myrecord.m_timetype, cond::SYNCH_ANY ); 
    editor.setDescription( "New Tag" );
    payloadType = editor.payloadType();
    editor.insert( firstSinceTime, firstPayloadId );
    cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
    editor.flush( a.usertext );
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
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  cond::persistency::TransactionScope scope( m_session.transaction() );
  Record& myrecord=this->lookUpRecord(recordName);
  if( myrecord.m_isNewTag ) {
    cond::throwException(std::string("Cannot append to non-existing tag ") + myrecord.m_tag,
		   "PoolDBOutputService::appendSinceTime");  
  }
  std::string payloadType("");
  try{
    cond::persistency::IOVEditor editor = m_session.editIov( myrecord.m_tag ); 
    payloadType = editor.payloadType();
    editor.insert( time, payloadId );
    cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
    editor.flush( a.usertext );

  }catch(const std::exception& er){
    cond::throwException(std::string(er.what()),
		   "PoolDBOutputService::appendSinceTime");
  }
  scope.close();
}

cond::service::PoolDBOutputService::Record& 
cond::service::PoolDBOutputService::lookUpRecord(const std::string& recordName){
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  if (!m_dbstarted) this->initDB();
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

cond::UserLogInfo& 
cond::service::PoolDBOutputService::lookUpUserLogInfo(const std::string& recordName){
  std::map<std::string,cond::UserLogInfo>::iterator it=m_logheaders.find(recordName);
  if(it==m_logheaders.end()) throw cond::Exception("Log db was not set for record " + recordName + " from PoolDBOutputService::lookUpUserLogInfo");
  return it->second;
}

void 
cond::service::PoolDBOutputService::closeIOV(Time_t lastTill, const std::string& recordName, 
					     bool withlogging) {
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
  // not fully working.. not be used for now...
  Record & myrecord  = lookUpRecord(recordName);
  cond::persistency::TransactionScope scope( m_session.transaction() );

  if( myrecord.m_isNewTag ) {
    cond::throwException(std::string("Cannot close non-existing tag ") + myrecord.m_tag,
			 "PoolDBOutputService::closeIOV");
  }
  cond::persistency::IOVEditor editor = m_session.editIov( myrecord.m_tag ); 
  editor.setEndOfValidity( lastTill );
  editor.flush("Tag closed.");
  scope.close();
}


void
cond::service::PoolDBOutputService::setLogHeaderForRecord(const std::string& recordName,const std::string& dataprovenance,const std::string& usertext)
{
  cond::UserLogInfo& myloginfo=this->lookUpUserLogInfo(recordName);
  myloginfo.provenance=dataprovenance;
  myloginfo.usertext=usertext;
}

// Still required.
void 
cond::service::PoolDBOutputService::tagInfo(const std::string& recordName,cond::TagInfo_t& result ){
  //
  std::lock_guard<std::recursive_mutex> lock(m_mutex);
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
