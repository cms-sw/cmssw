#include "CondCore/ORA/interface/DatabaseUtility.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Handle.h"
#include "DatabaseUtilitySession.h"
#include "DatabaseContainer.h"

ora::DatabaseUtility::DatabaseUtility():
  m_session(){
}

ora::DatabaseUtility::DatabaseUtility( Handle<DatabaseUtilitySession>& utilitySession ):
  m_session( utilitySession ){
}

ora::DatabaseUtility::DatabaseUtility( const DatabaseUtility& rhs ):
  m_session( rhs.m_session ){
}

ora::DatabaseUtility::~DatabaseUtility(){
}

ora::DatabaseUtility& ora::DatabaseUtility::operator=( const DatabaseUtility& rhs ){
  m_session = rhs.m_session;
  return *this;
}

std::set<std::string> ora::DatabaseUtility::listMappingVersions( const std::string& containerName ){
  Handle<DatabaseContainer> cont = m_session->containerHandle( containerName );
  if( !cont ){
    throwException("Container \""+containerName+"\" does not exist in the database.",
                   "DatabaseUtility::listMappingVersions");
  }
  return m_session->listMappingVersions( cont->id() );
}

std::map<std::string,std::string> ora::DatabaseUtility::listMappings( const std::string& containerName ){
  Handle<DatabaseContainer> cont = m_session->containerHandle( containerName );
  if( !cont ){
    throwException("Container \""+containerName+"\" does not exist in the database.",
                   "DatabaseUtility::listMappings");
  }
  return m_session->listMappings( cont->id() );  
}

bool ora::DatabaseUtility::dumpMapping( const std::string& mappingVersion,
                                        std::ostream& outputStream ){
  return m_session->dumpMapping( mappingVersion,outputStream );
}

void ora::DatabaseUtility::importContainerSchema( const std::string& sourceConnectionString,
                                                  const std::string& containerName ){
  m_session->importContainerSchema( sourceConnectionString, containerName );
}

void ora::DatabaseUtility::importContainer( const std::string& sourceConnectionString,
                                            const std::string& containerName ){
  m_session->importContainer( sourceConnectionString, containerName );
}

void ora::DatabaseUtility::eraseMapping( const std::string& mappingVersion ){
  m_session->eraseMapping( mappingVersion );
}

