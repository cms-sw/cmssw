#include "CondCore/ORA/interface/DatabaseUtility.h"
#include "CondCore/ORA/interface/Transaction.h"
#include "CondCore/ORA/interface/Exception.h"
#include "CondCore/ORA/interface/Handle.h"
#include "DatabaseUtilitySession.h"

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
  return m_session->listMappingVersions( containerName );
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

