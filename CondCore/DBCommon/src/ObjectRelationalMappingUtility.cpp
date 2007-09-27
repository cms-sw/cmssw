#include "CondCore/DBCommon/interface/ObjectRelationalMappingUtility.h"
#include "CondCore/DBCommon/interface/RelationalStorageManager.h"
#include "ObjectRelationalAccess/ObjectRelationalMappingUtilities.h"
//#include "ObjectRelationalAccess/ObjectRelationalException.h"
//#include "RelationalAccess/ISessionProxy.h"
cond::ObjectRelationalMappingUtility::ObjectRelationalMappingUtility( cond::RelationalStorageManager& coraldb ){
  m_mappingutil=new pool::ObjectRelationalMappingUtilities( &(coraldb.sessionProxy() ));
}
cond::ObjectRelationalMappingUtility::~ObjectRelationalMappingUtility(){
  delete m_mappingutil;
}
void cond::ObjectRelationalMappingUtility::buildAndStoreMappingFromBuffer( const std::string& buffer ){
  m_mappingutil->buildAndStoreMappingFromBuffer( buffer.c_str() );
}
void cond::ObjectRelationalMappingUtility::listMappings( std::vector<std::string>& mappinglist ){
  mappinglist=m_mappingutil->listMappings();
}
bool cond::ObjectRelationalMappingUtility::existsMapping(const std::string& version){
  return m_mappingutil->existsMapping(version);
}
void cond::ObjectRelationalMappingUtility::removeMapping(const std::string& version, bool removeDataTables){
  m_mappingutil->removeMapping(version,removeDataTables);
}
