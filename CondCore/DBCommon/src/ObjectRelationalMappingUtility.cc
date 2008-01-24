#include "CondCore/DBCommon/interface/ObjectRelationalMappingUtility.h"
#include "ObjectRelationalAccess/ObjectRelationalMappingUtilities.h"
#include <cassert>
cond::ObjectRelationalMappingUtility::ObjectRelationalMappingUtility( coral::ISessionProxy* coralsessionHandle ){
  m_mappingutil=new pool::ObjectRelationalMappingUtilities( coralsessionHandle );
}
cond::ObjectRelationalMappingUtility::~ObjectRelationalMappingUtility(){
  delete m_mappingutil;
}
void cond::ObjectRelationalMappingUtility::buildAndStoreMappingFromBuffer( const std::string& buffer ){
  // The following is temporarily commented out (and replaced with the
  // assert) pending migration to POOL_2_7_0
  //m_mappingutil->buildAndStoreMappingFromBuffer( buffer.c_str() );
  assert(0); 
}
void cond::ObjectRelationalMappingUtility::listMappings( std::vector<std::string>& mappinglist ){
  // The following is temporarily commented out (and replaced with the
  // assert) pending migration to POOL_2_7_0
  //mappinglist=m_mappingutil->listMappings();
  assert(0); 
}
bool cond::ObjectRelationalMappingUtility::existsMapping(const std::string& version){
  return m_mappingutil->existsMapping(version);
}
void cond::ObjectRelationalMappingUtility::removeMapping(const std::string& version){
  m_mappingutil->removeMapping(version);
}
