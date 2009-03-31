#include "CondCore/DBCommon/interface/ObjectRelationalMappingUtility.h"
#include "ObjectRelationalAccess/ObjectRelationalMappingUtilities.h"
#include "ObjectRelationalAccess/ObjectRelationalMappingPersistency.h"
#include "ObjectRelationalAccess/ObjectRelationalMappingPersistency.h"
#include "ObjectRelationalAccess/ObjectRelationalMappingPersistency.h"
#include "ObjectRelationalAccess/ObjectRelationalMappingSchema.h"
#include "RelationalAccess/ISessionProxy.h"
cond::ObjectRelationalMappingUtility::ObjectRelationalMappingUtility( coral::ISessionProxy* coralsessionHandle ):m_coralsessionHandle(coralsessionHandle){
  m_mappingutil=new pool::ObjectRelationalMappingUtilities( coralsessionHandle );
}

cond::ObjectRelationalMappingUtility::~ObjectRelationalMappingUtility(){
  delete m_mappingutil;
}

void cond::ObjectRelationalMappingUtility::buildAndStoreMappingFromBuffer( const std::string& buffer ){
  m_mappingutil->buildAndMaterializeMappingFromBuffer( buffer.c_str(),false,false );
}

void cond::ObjectRelationalMappingUtility::buildAndStoreMappingFromFile( const std::string& filename ){
  m_mappingutil->buildAndMaterializeMapping( filename,"",false,false );
}

/*void cond::ObjectRelationalMappingUtility::listMappings( std::vector<std::string>& mappinglist ){
  mappinglist=m_mappingutil->listMappings();
}
*/

bool cond::ObjectRelationalMappingUtility::existsMapping(const std::string& version){
  pool::ObjectRelationalMappingSchema mappingSchema(m_coralsessionHandle->nominalSchema());
  if(!mappingSchema.existTables()) return false;
  return m_mappingutil->existsMapping(version);
}

void cond::ObjectRelationalMappingUtility::removeMapping(const std::string& version,bool removeTables){
  m_mappingutil->removeMapping(version,removeTables);
}


bool 
cond::ObjectRelationalMappingUtility::exportMapping(coral::ISessionProxy* session, 
						    std::string const & contName, std::string const & classVersion, 
						    bool allVersions=false) {
  m_mappingutil->loadMappingInformation( contName, classVersion, allVersions);
  m_mappingutil->setSession(session);
  bool okStore = m_mappingutil->storeMappingInformation();
  m_mappingutil->setSession(m_coralsessionHandle);
  retun okStore;
}
