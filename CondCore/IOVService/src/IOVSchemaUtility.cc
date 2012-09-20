#include "CondCore/DBCommon/interface/DbSession.h"
#include "CondCore/DBCommon/interface/Exception.h"
#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/IOVService/interface/IOVNames.h"

cond::IOVSchemaUtility::IOVSchemaUtility(cond::DbSession& session):
  m_session( session ),
  m_log(0){
}

cond::IOVSchemaUtility::IOVSchemaUtility(cond::DbSession& session, std::ostream& log):
  m_session( session ),
  m_log(&log){
}
cond::IOVSchemaUtility::~IOVSchemaUtility(){}

bool cond::IOVSchemaUtility::createIOVContainerIfNecessary(){
  ora::Database& db = m_session.storage();
  if( !db.exists() ){
    if(m_log) *m_log << "INFO: Creating condition database in "<<db.connectionString()<<std::endl;
    db.create(cond::DbSession::COND_SCHEMA_VERSION);
    db.setAccessPermission(cond::DbSession::CONDITIONS_GENERAL_READER, false );
    db.setAccessPermission( cond::DbSession::CONDITIONS_GENERAL_WRITER, true );
  } 
  std::set<std::string> conts = db.containers();
  if( conts.find( cond::IOVNames::container() )==conts.end() ){
    if(m_log) *m_log << "INFO: Creating container \"" << cond::IOVNames::container() << "\" in "<<db.connectionString()<<std::endl;
    ora::Container c = db.createContainer( cond::IOVNames::container(), cond::IOVNames::container() );
    c.setAccessPermission( cond::DbSession::CONDITIONS_GENERAL_READER, false );
    c.setAccessPermission( cond::DbSession::CONDITIONS_GENERAL_WRITER, true );
    return true;
  }
  if(m_log) *m_log << "INFO: container \"" << cond::IOVNames::container() << "\" already exists in the database "<<db.connectionString()<<std::endl;
  return false;
}

bool cond::IOVSchemaUtility::dropIOVContainer(){
  ora::Database& db = m_session.storage();
  std::set<std::string> conts = db.containers();
  if( conts.find( cond::IOVNames::container() )==conts.end() ){
    if(m_log) *m_log << "WARNING: container \"" << cond::IOVNames::container() << "\" does not exist in the database "<<db.connectionString()<<std::endl;
    return false;
  }
  if(m_log) *m_log << "INFO: Dropping container \"" << cond::IOVNames::container() << "\" from "<<db.connectionString()<<std::endl;
  db.dropContainer( cond::IOVNames::container() );
  return true;
}

void cond::IOVSchemaUtility::createPayloadContainer( const std::string& payloadName, 
						     const std::string& payloadTypeName ){
  ora::Database& db = m_session.storage();
  std::set<std::string> conts = db.containers();
  if( conts.find( payloadName ) != conts.end()) throw cond::Exception("Container \""+payloadName+"\" already exists.");
  if(m_log) *m_log << "INFO: Creating container \"" << payloadName << "\" in "<<db.connectionString()<<std::endl;
  ora::Container c = db.createContainer( payloadTypeName, payloadName );
  c.setAccessPermission( cond::DbSession::CONDITIONS_GENERAL_READER, false );
  c.setAccessPermission( cond::DbSession::CONDITIONS_GENERAL_WRITER, true );
}

void cond::IOVSchemaUtility::dropPayloadContainer( const std::string& payloadName ){
  ora::Database& db = m_session.storage();
  std::set<std::string> conts = db.containers();
  if( conts.find( payloadName )!=conts.end() ){
    if(m_log) *m_log << "INFO: Dropping container \"" << payloadName << "\" from "<<db.connectionString()<<std::endl;
    db.dropContainer( payloadName );
    return;
  } 
  if(m_log) *m_log << "WARNING: container \"" << payloadName << "\" does not exist in the database "<<db.connectionString()<<std::endl;
}

void cond::IOVSchemaUtility::dropAll(){
  ora::Database& db = m_session.storage();
  if(m_log) *m_log << "INFO: Dropping database in "<<db.connectionString()<<std::endl;
  db.drop();
}

