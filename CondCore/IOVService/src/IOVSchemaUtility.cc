#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/IOVService/interface/IOVNames.h"
#include "CondCore/DBCommon/interface/ObjectRelationalMappingUtility.h"
#include "CondCore/DBCommon/interface/CoralTransaction.h"
cond::IOVSchemaUtility::IOVSchemaUtility(cond::CoralTransaction& coraldb):m_coraldb(coraldb){
}
void 
cond::IOVSchemaUtility::create(){
  cond::ObjectRelationalMappingUtility mappingUtil(&(m_coraldb.coralSessionProxy()) );
  if( !mappingUtil.existsMapping(cond::IOVNames::iovMappingVersion()) ){
    mappingUtil.buildAndStoreMappingFromBuffer(cond::IOVNames::iovMappingXML());
  }
}
void 
cond::IOVSchemaUtility::drop(){
  cond::ObjectRelationalMappingUtility mappingUtil(&(m_coraldb.coralSessionProxy()) );
  if( !mappingUtil.existsMapping(cond::IOVNames::iovMappingVersion()) ) return;
  mappingUtil.removeMapping(cond::IOVNames::iovMappingVersion(),true);
}
void
cond::IOVSchemaUtility::truncate(){
  cond::ObjectRelationalMappingUtility mappingUtil(&(m_coraldb.coralSessionProxy()) );
  if( !mappingUtil.existsMapping(cond::IOVNames::iovMappingVersion()) ) return;
  mappingUtil.removeMapping(cond::IOVNames::iovMappingVersion(),false);
}
