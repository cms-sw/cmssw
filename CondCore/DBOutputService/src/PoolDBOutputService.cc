#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/Exception.h"
#include "CondCore/DBCommon/interface/DbConnection.h"
#include "CondCore/DBCommon/interface/DbOpenTransaction.h"
#include "CondCore/DBCommon/interface/DbTransaction.h"
#include "CondCore/DBCommon/interface/TagInfo.h"
#include "CondCore/DBCommon/interface/IOVInfo.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVProxy.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <vector>
#include<memory>

void 
cond::service::PoolDBOutputService::fillRecord( edm::ParameterSet & pset) {
  Record thisrecord;

  thisrecord.m_idName = pset.getParameter<std::string>("record");
  thisrecord.m_tag = pset.getParameter<std::string>("tag");
  
  thisrecord.m_closeIOV =
    pset.getUntrackedParameter<bool>("closeIOV", m_closeIOV);
 
  thisrecord.m_freeInsert = 
    pset.getUntrackedParameter<bool>("outOfOrder",m_freeInsert);
  
  thisrecord.m_timetype=cond::findSpecs(pset.getUntrackedParameter< std::string >("timetype",m_timetypestr)).type;

  m_callbacks.insert(std::make_pair(thisrecord.m_idName,thisrecord));
 
  if( !m_logConnectionString.empty() ){
      cond::UserLogInfo userloginfo;
      m_logheaders.insert(std::make_pair(thisrecord.m_idName,userloginfo));
  }
}

cond::service::PoolDBOutputService::PoolDBOutputService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & iAR ): 
  m_timetypestr(""),
  m_currentTime( 0 ),
  m_connectionString(""),
  m_session(),
  m_logConnectionString(""),
  m_logdb(),
  m_dbstarted( false ),
  m_callbacks(),
  m_newtags(),
  m_closeIOV(false),
  m_freeInsert(false),
  m_logheaders()
{
  m_closeIOV=iConfig.getUntrackedParameter<bool>("closeIOV",m_closeIOV);

  if( iConfig.exists("outOfOrder") ){
     m_freeInsert=iConfig.getUntrackedParameter<bool>("outOfOrder");
  }  

  m_timetypestr=iConfig.getUntrackedParameter< std::string >("timetype","runnumber");
  m_timetype=cond::findSpecs( m_timetypestr).type;

  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters");
  cond::DbConnection connection;
  connection.configuration().setParameters( connectionPset );
  connection.configure();

  m_connectionString = iConfig.getParameter<std::string>("connect");
  m_session = connection.createSession();
  m_session.open( m_connectionString, Auth::COND_WRITER_ROLE );  
  
  if( iConfig.exists("logconnect") ){
    m_logConnectionString = iConfig.getUntrackedParameter<std::string>("logconnect");
    cond::DbSession logSession = connection.createSession();
    m_logdb.reset( new cond::Logger( logSession ) );
  }  
  
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

cond::DbSession
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
  DbOpenTransaction trans( m_session.transaction() );
  try{ 
    if(!forReading) {
      cond::IOVSchemaUtility schemaUtil( m_session );
      schemaUtil.createIOVContainer();
      m_session.storage().lockContainer( IOVNames::container() );
    }
    //init logdb if required
    if(!m_logConnectionString.empty()){
      m_logdb->connect( m_logConnectionString );
      m_logdb->createLogDBIfNonExist();
    }
  } catch( const std::exception& er ){
    throw cond::Exception( std::string(er.what()) + " from PoolDBOutputService::initDB" );
  }
  trans.ok();
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
cond::service::PoolDBOutputService::createNewIOV( GetToken const & payloadToken, 
                                                  cond::Time_t firstSinceTime, 
                                                  cond::Time_t firstTillTime,
                                                  const std::string& recordName, 
                                                  bool withlogging){
  DbOpenTransaction trans( m_session.transaction() );
  Record& myrecord=this->lookUpRecord(recordName);
  if(!myrecord.m_isNewTag) {
    throw cond::Exception(myrecord.m_tag + " is not a new tag from PoolDBOutputService::createNewIOV");
  }
  std::string iovToken;
  if(withlogging){
    if( m_logConnectionString.empty() ) {
       throw cond::Exception("Log db was not set from PoolDBOutputService::createNewIOV");
    }
  }
 
  std::string objToken;
  std::string objClass;
  unsigned int payloadIdx=0;
  try{
    cond::IOVEditor editor(m_session);
    editor.create(myrecord.m_timetype, firstTillTime);
    objToken = payloadToken(m_session);
    objClass = m_session.classNameForItem( objToken );
    unsigned int payloadIdx=editor.append(firstSinceTime, objToken);
    iovToken=editor.token();
    editor.stamp(cond::userInfo(),false);
    editor.setScope( cond::IOVSequence::Tag );
    
    cond::MetaData metadata(m_session);

    metadata.addMapping(myrecord.m_tag,iovToken,myrecord.m_timetype);

    m_newtags.push_back( std::pair<std::string,std::string>(myrecord.m_tag,iovToken) );
    myrecord.m_iovtoken=iovToken;
    myrecord.m_isNewTag=false;
    if(withlogging){
      std::string destconnect=m_session.connectionString();
      cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
      m_logdb->logOperationNow(a,destconnect,objClass,objToken,myrecord.m_tag,myrecord.timetypestr(),payloadIdx,firstSinceTime);
    }
  }catch(const std::exception& er){ 
    if(withlogging){
      std::string destconnect=m_session.connectionString();
      cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
      m_logdb->logFailedOperationNow(a,destconnect,objClass,objToken,myrecord.m_tag,myrecord.timetypestr(),payloadIdx,firstSinceTime,std::string(er.what()));
    }
    throw cond::Exception(std::string(er.what()) + " from PoolDBOutputService::createNewIOV ");
  }
  trans.ok();
}


void 
cond::service::PoolDBOutputService::add( GetToken const & payloadToken,  
					 cond::Time_t time,
					 const std::string& recordName,
					 bool withlogging) {
  DbOpenTransaction trans( m_session.transaction() );
  Record& myrecord=this->lookUpRecord(recordName);
  if(withlogging){
    if( m_logConnectionString.empty() ) {
       throw cond::Exception("Log db was not set from PoolDBOutputService::add");
    }
  }

  std::string objToken;
  std::string objClass;
  unsigned int payloadIdx=0;

  try{
    objToken = payloadToken(m_session);
    objClass = m_session.classNameForItem( objToken );
    payloadIdx= appendIOV(m_session,myrecord,objToken,time);
    if(withlogging){
      std::string destconnect=m_session.connectionString();
      cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
      m_logdb->logOperationNow(a,destconnect,objClass,objToken,myrecord.m_tag,myrecord.timetypestr(),payloadIdx,time);
    }
  }catch(const std::exception& er){
    if(withlogging){
      std::string destconnect=m_session.connectionString();
      cond::UserLogInfo a=this->lookUpUserLogInfo(recordName);
      m_logdb->logFailedOperationNow(a,destconnect,objClass,objToken,myrecord.m_tag,myrecord.timetypestr(),payloadIdx,time,std::string(er.what()));
    }
    throw cond::Exception(std::string(er.what()) + " from PoolDBOutputService::add ");
  }
  trans.ok();
}

cond::service::PoolDBOutputService::Record& 
cond::service::PoolDBOutputService::lookUpRecord(const std::string& recordName){
  if (!m_dbstarted) this->initDB( false );
  DbOpenTransaction trans( m_session.transaction() );
  std::map<std::string,Record>::iterator it=m_callbacks.find(recordName);
  if(it==m_callbacks.end()) {
    throw cond::UnregisteredRecordException(recordName + " from PoolDBOutputService::lookUpRecord");
  }
  cond::MetaData metadata(m_session);
  if( !metadata.hasTag(it->second.m_tag) ){
    it->second.m_iovtoken="";
    it->second.m_isNewTag=true;
  }else{
    it->second.m_iovtoken=metadata.getToken(it->second.m_tag);
    it->second.m_isNewTag=false;
  }
  trans.ok();
  return it->second;
}
cond::UserLogInfo& 
cond::service::PoolDBOutputService::lookUpUserLogInfo(const std::string& recordName){
  std::map<std::string,cond::UserLogInfo>::iterator it=m_logheaders.find(recordName);
  if(it==m_logheaders.end()) throw cond::Exception("Log db was not set for record " + recordName + " from PoolDBOutputService::lookUpUserLogInfo");
  return it->second;
}


unsigned int 
cond::service::PoolDBOutputService::appendIOV(cond::DbSession& pooldb,
						   Record& record, 
						   const std::string& payloadToken, 
						   cond::Time_t sinceTime){
  DbOpenTransaction trans( m_session.transaction() );
  if( record.m_isNewTag ) {
    throw cond::Exception(std::string("Cannot append to non-existing tag ") + record.m_tag + std::string(" from PoolDBOutputService::appendIOV"));  
  }

  cond::IOVEditor editor(pooldb,record.m_iovtoken);
  
  unsigned int payloadIdx =  record.m_freeInsert ? 
    editor.freeInsert(sinceTime,payloadToken) :
    editor.append(sinceTime,payloadToken);
  if (record.m_closeIOV) editor.updateClosure(sinceTime);
  editor.stamp(cond::userInfo(),false);
  trans.ok();
  return payloadIdx;
}

void 
cond::service::PoolDBOutputService::closeIOV(Time_t lastTill, const std::string& recordName, 
					     bool withlogging) {
  // not fully working.. not be used for now...
  Record & record  = lookUpRecord(recordName);
  DbOpenTransaction trans( m_session.transaction() );

  if( record.m_isNewTag ) {
    throw cond::Exception(std::string("Cannot close non-existing tag ") + record.m_tag + std::string(" from PoolDBOutputService::closeIOV"));
  }
  cond::IOVEditor editor(m_session,record.m_iovtoken);
  editor.updateClosure(lastTill);
  editor.stamp(cond::userInfo(),false);
  trans.ok();
}



void
cond::service::PoolDBOutputService::setLogHeaderForRecord(const std::string& recordName,const std::string& dataprovenance,const std::string& usertext)
{
  cond::UserLogInfo& myloginfo=this->lookUpUserLogInfo(recordName);
  myloginfo.provenance=dataprovenance;
  myloginfo.usertext=usertext;
}


const cond::Logger& 
cond::service::PoolDBOutputService::queryLog()const{
  if( !m_logdb.get() ) throw cond::Exception("Log database is not set from PoolDBOutputService::queryLog");
  return *m_logdb;
}


void 
cond::service::PoolDBOutputService::tagInfo(const std::string& recordName,cond::TagInfo& result ){
  Record& record = lookUpRecord(recordName);
  result.name=record.m_tag;
  result.token=record.m_iovtoken;
  //use iovproxy to find out.
  cond::IOVProxy iov(m_session, record.m_iovtoken);
  result.size=iov.size();
  if (result.size>0) {
    // get last object
    iov.tail(1);
    cond::IOVElementProxy last = *iov.begin();
    result.lastInterval = cond::ValidityInterval(last.since(), last.till());
    result.lastPayloadToken=last.token();
  }
}
