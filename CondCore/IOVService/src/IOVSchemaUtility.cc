#include "CondCore/IOVService/interface/IOVSchemaUtility.h"
#include "CondCore/IOVService/interface/IOVNames.h"

cond::IOVSchemaUtility::IOVSchemaUtility(cond::DbSession& pooldb):m_pooldb(pooldb){
}
cond::IOVSchemaUtility::~IOVSchemaUtility(){}
void 
cond::IOVSchemaUtility::create(){
  //m_pooldb.initializeMapping( cond::IOVNames::iovMappingVersion(), cond::IOVNames::iovMappingXML());
}
void 
cond::IOVSchemaUtility::drop(){
  //m_pooldb.deleteMapping( cond::IOVNames::iovMappingVersion(), true );
}
void
cond::IOVSchemaUtility::truncate(){
  //m_pooldb.deleteMapping( cond::IOVNames::iovMappingVersion(), false );
}
