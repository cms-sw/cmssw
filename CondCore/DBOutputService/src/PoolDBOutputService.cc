#include "CondCore/DBOutputService/interface/PoolDBOutputService.h"
#include "CondCore/DBOutputService/interface/TagInfo.h"
#include "DataFormats/Provenance/interface/EventID.h"
#include "DataFormats/Provenance/interface/Timestamp.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "CondCore/DBCommon/interface/ConnectionHandler.h"
#include "CondCore/IOVService/interface/IOVService.h"
#include "CondCore/IOVService/interface/IOVEditor.h"
#include "CondCore/IOVService/interface/IOVIterator.h"
//#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
//#include "CondCore/DBCommon/interface/ConnectMode.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/DBCommon/interface/ConfigSessionFromParameterSet.h"
#include "CondCore/DBCommon/interface/SessionConfiguration.h"
#include "CondCore/DBOutputService/interface/Exception.h"
//#include "CondCore/DBCommon/interface/ObjectRelationalMappingUtility.h"
#include "CondCore/DBCommon/interface/DBSession.h"
//#include "FWCore/Framework/interface/IOVSyncValue.h"

//POOL include
//#include "FileCatalog/IFileCatalog.h"
#include "serviceCallbackToken.h"
#include "CondCore/DBOutputService/interface/UserLogInfo.h"
//#include <iostream>
#include <vector>

static cond::ConnectionHandler& conHandler=cond::ConnectionHandler::Instance();


cond::service::PoolDBOutputService::PoolDBOutputService(const edm::ParameterSet & iConfig,edm::ActivityRegistry & iAR ): 
  //m_currentTime( 0 ),
  m_session( 0 ),
  m_dbstarted( false ),
  m_logdb( 0 ),
  m_logdbOn( false )
{
  std::string connect=iConfig.getParameter<std::string>("connect");
  std::string logconnect("");
  if( iConfig.exists("logconnect") ){
    logconnect=iConfig.getUntrackedParameter<std::string>("logconnect");
  }  
  m_timetypestr=iConfig.getUntrackedParameter< std::string >("timetype","runnumber");
  if((m_timetypestr!=std::string("runnumber"))&&(m_timetypestr!=std::string("timestamp"))){
    throw cond::Exception(std::string("Unrecognised time type ")+m_timetypestr);
  }else{
    if(m_timetypestr==std::string("runnumber")) m_timetype=cond::runnumber;
    if(m_timetypestr==std::string("timestamp")) m_timetype=cond::timestamp;
  }
  m_session=new cond::DBSession;  
  std::string blobstreamerName("");
  if( iConfig.exists("BlobStreamerName") ){
    blobstreamerName=iConfig.getUntrackedParameter<std::string>("BlobStreamerName");
    blobstreamerName.insert(0,"COND/Services/");
    m_session->configuration().setBlobStreamer(blobstreamerName);
  }
  edm::ParameterSet connectionPset = iConfig.getParameter<edm::ParameterSet>("DBParameters"); 
  ConfigSessionFromParameterSet configConnection(*m_session,connectionPset);
  //std::string catconnect("pfncatalog_memory://POOL_RDBMS?");
  //catconnect.append(connect);
  conHandler.registerConnection("outputdb",connect,-1);
  if( !logconnect.empty() ){
    m_logdbOn=true;
    conHandler.registerConnection("logdb",logconnect,-1);
  }
  m_session->open();
  conHandler.connect(m_session);
  m_connection=conHandler.getConnection("outputdb");
  typedef std::vector< edm::ParameterSet > Parameters;
  Parameters toPut=iConfig.getParameter<Parameters>("toPut");
  for(Parameters::iterator itToPut = toPut.begin(); itToPut != toPut.end(); ++itToPut) {
    cond::service::serviceCallbackRecord thisrecord;
    thisrecord.m_containerName = itToPut->getParameter<std::string>("record");
    thisrecord.m_tag = itToPut->getParameter<std::string>("tag");
    m_callbacks.insert(std::make_pair(cond::service::serviceCallbackToken::build(thisrecord.m_containerName),thisrecord));
    if(m_logdbOn){
      cond::service::UserLogInfo userloginfo;
      m_logheaders.insert(std::make_pair(cond::service::serviceCallbackToken::build(thisrecord.m_containerName),userloginfo));
    }
  }
  iAR.watchPreProcessEvent(this,&cond::service::PoolDBOutputService::preEventProcessing);
  iAR.watchPostEndJob(this,&cond::service::PoolDBOutputService::postEndJob);
  iAR.watchPreModule(this,&cond::service::PoolDBOutputService::preModule);
  iAR.watchPostModule(this,&cond::service::PoolDBOutputService::postModule);
}
cond::Connection& 
cond::service::PoolDBOutputService::connection() const{
  return *m_connection;
}
std::string 
cond::service::PoolDBOutputService::tag( const std::string& EventSetupRecordName ){
  return this->lookUpRecord(EventSetupRecordName).m_tag;
}

bool 
cond::service::PoolDBOutputService::isNewTagRequest( const std::string& EventSetupRecordName ){
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if(!m_dbstarted) this->initDB();
  return myrecord.m_isNewTag;
}
void 
cond::service::PoolDBOutputService::initDB()
{
  if(m_dbstarted) return;
  cond::CoralTransaction& coraldb=m_connection->coralTransaction();
  try{
    coraldb.start(false);
    IOVSchemaUtility ut(coraldb);
    ut.create();
    /*cond::ObjectRelationalMappingUtility* mappingUtil=new cond::ObjectRelationalMappingUtility(&(coraldb.coralSessionProxy()) );
      if( !mappingUtil->existsMapping(cond::IOVNames::iovMappingVersion()) ){
      mappingUtil->buildAndStoreMappingFromBuffer(cond::IOVNames::iovMappingXML());
      }
      delete mappingUtil;
    */
    cond::MetaData metadata(coraldb);
    for(std::map<size_t,cond::service::serviceCallbackRecord>::iterator it=m_callbacks.begin(); it!=m_callbacks.end(); ++it){
      //std::string iovtoken;
      if( !metadata.hasTag(it->second.m_tag) ){
	it->second.m_iovtoken="";
	it->second.m_isNewTag=true;
      }else{
	it->second.m_iovtoken=metadata.getToken(it->second.m_tag);
	it->second.m_isNewTag=false;
      }
    }
    coraldb.commit();    
    //init logdb if required
    if(m_logdbOn){
      m_logdb=new cond::Logger(conHandler.getConnection("logdb"));
      m_logdb->getWriteLock();
      m_logdb->createLogDBIfNonExist();
      m_logdb->releaseWriteLock();
    }
  }catch( const std::exception& er ){
    throw cond::Exception( "PoolDBOutputService::initDB "+std::string(er.what()) );
  }
  m_dbstarted=true;
}

void 
cond::service::PoolDBOutputService::postEndJob()
{
  if(m_logdb){
    delete m_logdb;
  }
}

void 
cond::service::PoolDBOutputService::preEventProcessing(const edm::EventID& iEvtid, const edm::Timestamp& iTime)
{
  if( m_timetype == cond::runnumber ){
    m_currentTime=iEvtid.run();
  }else{ //timestamp
    m_currentTime=iTime.value();
  }
}
void
cond::service::PoolDBOutputService::preModule(const edm::ModuleDescription& desc){
}

void
cond::service::PoolDBOutputService::postModule(const edm::ModuleDescription& desc){
}

cond::service::PoolDBOutputService::~PoolDBOutputService(){
  delete m_session;
}

size_t 
cond::service::PoolDBOutputService::callbackToken(const std::string& EventSetupRecordName ) const {
  return cond::service::serviceCallbackToken::build(EventSetupRecordName);
}

cond::Time_t 
cond::service::PoolDBOutputService::endOfTime() const{
  switch(m_timetype){
  case cond::runnumber:
    return (cond::Time_t)edm::RunID::maxRunNumber();
  case cond::timestamp:  
    return (cond::Time_t)edm::Timestamp::endOfTime().value();
  default:
    return (cond::Time_t)edm::Timestamp::endOfTime().value();
  }
}
cond::Time_t 
cond::service::PoolDBOutputService::beginOfTime() const{
  switch(m_timetype){ 
  case cond::runnumber:
    return (cond::Time_t)edm::RunID::firstValidRun().run();
  case cond::timestamp:
    return (cond::Time_t)edm::Timestamp::beginOfTime().value();
  default:
    return (cond::Time_t)edm::Timestamp::beginOfTime().value();
  }
}
cond::Time_t 
cond::service::PoolDBOutputService::currentTime() const{
  return m_currentTime;
}

void 
cond::service::PoolDBOutputService::createNewIOV( GetToken const & payloadToken, cond::Time_t firstSinceTime, cond::Time_t firstTillTime,const std::string& EventSetupRecordName, bool withlogging){
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if (!m_dbstarted) this->initDB();
  if(!myrecord.m_isNewTag) throw cond::Exception("PoolDBOutputService::createNewIO not a new tag");
  cond::PoolTransaction& pooldb=m_connection->poolTransaction();
  std::string iovToken;
  if(withlogging){
    m_logdb->getWriteLock();
  }
 
  std::string objToken;
  unsigned int payloadIdx=0;
  try{
    pooldb.start(false);

    cond::IOVService iovmanager(pooldb,m_timetype);
    cond::IOVEditor* editor=iovmanager.newIOVEditor("");
    editor->create(firstSinceTime,iovmanager.timeType());
    objToken = payloadToken(pooldb);
    unsigned int payloadIdx=editor->insert(firstTillTime, objToken);
    iovToken=editor->token();
    delete editor;    

    pooldb.commit();

    cond::CoralTransaction& coraldb=m_connection->coralTransaction();
    cond::MetaData metadata(coraldb);
    coraldb.start(false);
    /*
    MetaDataEntry imetadata;
    imetadata.tagname=myrecord.m_tag;
    imetadata.iovtoken=iovToken;
    imetadata.timetype=m_timetype;
    imetadata.firstsince=firstSinceTime;
    metadata.addMapping(imetadata);
   */
    metadata.addMapping(myrecord.m_tag,iovToken,m_timetype);
    coraldb.commit();
    m_newtags.push_back( std::make_pair<std::string,std::string>(myrecord.m_tag,iovToken) );
    myrecord.m_iovtoken=iovToken;
    myrecord.m_isNewTag=false;
    if(withlogging){
      if(!m_logdb)throw cond::Exception("cannot log to non-existing log db");
      std::string destconnect=m_connection->connectStr();
      cond::service::UserLogInfo a=this->lookUpUserLogInfo(EventSetupRecordName);
      m_logdb->logOperationNow(a,destconnect,objToken,myrecord.m_tag,m_timetypestr,payloadIdx);
    }
  }catch(const std::exception& er){ 
    if(withlogging){
      std::string destconnect=m_connection->connectStr();
      cond::service::UserLogInfo a=this->lookUpUserLogInfo(EventSetupRecordName);
      m_logdb->logFailedOperationNow(a,destconnect,objToken,myrecord.m_tag,m_timetypestr,payloadIdx,std::string(er.what()));
      m_logdb->releaseWriteLock();
    }
    throw cond::Exception("PoolDBOutputService::createNewIOV "+std::string(er.what()));
  }
  if(withlogging){
    m_logdb->releaseWriteLock();
  }
}


void 
cond::service::PoolDBOutputService::add( bool sinceNotTill, 
					 GetToken const & payloadToken,  
					 cond::Time_t time,
					 const std::string& EventSetupRecordName,
					 bool withlogging) {
  cond::service::serviceCallbackRecord& myrecord=this->lookUpRecord(EventSetupRecordName);
  if (!m_dbstarted) this->initDB();
  cond::PoolTransaction& pooldb=m_connection->poolTransaction();
  if(withlogging){
    m_logdb->getWriteLock();
  }

  std::string objToken;
  unsigned int payloadIdx=0;

  try{
    pooldb.start(false);
    objToken = payloadToken(pooldb);
    payloadIdx= sinceNotTill ?
      this->appendIOV(pooldb,myrecord,objToken,time) :
      this->insertIOV(pooldb,myrecord,objToken,time);

    pooldb.commit();
    if(withlogging){
      if(!m_logdb)throw cond::Exception("cannot log to non-existing log db");
      std::string destconnect=m_connection->connectStr();
      cond::service::UserLogInfo a=this->lookUpUserLogInfo(EventSetupRecordName);
      m_logdb->logOperationNow(a,destconnect,objToken,myrecord.m_tag,m_timetypestr,payloadIdx);
    }
  }catch(const std::exception& er){
    if(withlogging){
      if(!m_logdb)throw cond::Exception("cannot log to non-existing log db");
      std::string destconnect=m_connection->connectStr();
      cond::service::UserLogInfo a=this->lookUpUserLogInfo(EventSetupRecordName);
      m_logdb->logFailedOperationNow(a,destconnect,objToken,myrecord.m_tag,m_timetypestr,payloadIdx,std::string(er.what()));
      m_logdb->releaseWriteLock();
    }
    throw cond::Exception("PoolDBOutputService::add "+std::string(er.what()));
  }
  if(withlogging){
    m_logdb->releaseWriteLock();
  }
}

cond::service::serviceCallbackRecord& 
cond::service::PoolDBOutputService::lookUpRecord(const std::string& EventSetupRecordName){
  size_t callbackToken=this->callbackToken( EventSetupRecordName );
  std::map<size_t,cond::service::serviceCallbackRecord>::iterator it=m_callbacks.find(callbackToken);
  if(it==m_callbacks.end()) throw cond::UnregisteredRecordException(EventSetupRecordName);
  return it->second;
}

cond::service::UserLogInfo& 
cond::service::PoolDBOutputService::lookUpUserLogInfo(const std::string& EventSetupRecordName){
  size_t callbackToken=this->callbackToken( EventSetupRecordName );
  std::map<size_t,cond::service::UserLogInfo>::iterator it=m_logheaders.find(callbackToken);
  if(it==m_logheaders.end()) throw cond::UnregisteredRecordException(EventSetupRecordName);
  return it->second;
}


unsigned int 
cond::service::PoolDBOutputService::appendIOV(cond::PoolTransaction& pooldb,
						   cond::service::serviceCallbackRecord& record, 
						   const std::string& payloadToken, 
						   cond::Time_t sinceTime){
  if( record.m_isNewTag ) {
    throw cond::Exception(std::string("PoolDBOutputService::appendIOV: cannot append to non-existing tag ")+record.m_tag );  
  }

  cond::IOVService iovmanager(pooldb);  
  cond::IOVEditor* editor=iovmanager.newIOVEditor(record.m_iovtoken);
  unsigned int payloadIdx=editor->append(sinceTime,payloadToken);
  delete editor;
  return payloadIdx;
}

unsigned int
cond::service::PoolDBOutputService::insertIOV( cond::PoolTransaction& pooldb,
					       cond::service::serviceCallbackRecord& record, 
					       const std::string& payloadToken,
					       cond::Time_t tillTime){
  
  if( record.m_isNewTag ) {
    throw cond::Exception(std::string("PoolDBOutputService::insertIOV: cannot append to non-existing tag ")+record.m_tag );  
  }
  
  cond::IOVService iovmanager(pooldb);
  cond::IOVEditor* editor=iovmanager.newIOVEditor(record.m_iovtoken);
  unsigned int payloadIdx=editor->insert(tillTime,payloadToken);
  delete editor;    
  return payloadIdx;
}



void
cond::service::PoolDBOutputService::setLogHeaderForRecord(const std::string& EventSetupRecordName,const std::string& dataprovenance,const std::string& usertext)
{
  cond::service::UserLogInfo& myloginfo=this->lookUpUserLogInfo(EventSetupRecordName);
  myloginfo.provenance=dataprovenance;
  myloginfo.usertext=usertext;
}
const cond::Logger& 
cond::service::PoolDBOutputService::queryLog()const{
  if(!m_logdb) throw cond::Exception("PoolDBOutputService::queryLog ERROR: logging is off");
  return *m_logdb;
}


void 
cond::service::PoolDBOutputService::tagInfo(const std::string& EventSetupRecordName,cond::TagInfo& result ){
  cond::service::serviceCallbackRecord& record=this->lookUpRecord(EventSetupRecordName);
  result.name=record.m_tag;
  result.token=record.m_iovtoken;
  //use ioviterator to find out.
  cond::PoolTransaction& pooldb=m_connection->poolTransaction();
  pooldb.start(true);
  cond::IOVService iovmanager( pooldb );
  cond::IOVIterator* iit=iovmanager.newIOVIterator(result.token,cond::IOVService::backwardIter);
  iit->next(); // just to initialize
  result.lastInterval=iit->validity();
  result.lastPayloadToken=iit->payloadToken();
  result.size=iit->size();
  pooldb.commit();
  delete iit;
 }
