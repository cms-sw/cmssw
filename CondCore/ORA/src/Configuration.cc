#include "CondCore/ORA/interface/Configuration.h"
#include "CondCore/ORA/interface/IBlobStreamingService.h"
#include "CondCore/ORA/interface/IReferenceHandler.h"

std::string ora::Configuration::automaticDatabaseCreation(){
  static std::string s_automaticDatabaseCreation("ORA_AUTOMATIC_DATABASE_CREATION");
  return s_automaticDatabaseCreation;  
}

std::string ora::Configuration::automaticContainerCreation(){
  static std::string s_automaticContainerCreation("ORA_AUTOMATIC_CONTAINER_CREATION");
  return s_automaticContainerCreation;  
}

std::string ora::Configuration::automaticSchemaEvolution(){
  static std::string s_automaticSchemaEvolution("ORA_AUTOMATIC_SCHEMA_EVOLUTION");
  return s_automaticSchemaEvolution;  
}

ora::Configuration::Configuration():
  m_blobStreamingService(),
  m_referenceHandler(),
  m_properties(){
  
  coral::MessageStream::setMsgVerbosity( coral::Info );

}

ora::Configuration::~Configuration(){
}

void ora::Configuration::setBlobStreamingService( IBlobStreamingService* service ){
  m_blobStreamingService.reset( service );
}

ora::IBlobStreamingService* ora::Configuration::blobStreamingService(){
  return m_blobStreamingService.get();
}

void ora::Configuration::setReferenceHandler( IReferenceHandler* handler ){
  m_referenceHandler.reset( handler );
}

ora::IReferenceHandler* ora::Configuration::referenceHandler(){
  return m_referenceHandler.get();
}

ora::Properties& ora::Configuration::properties(){
  return m_properties;
}

void ora::Configuration::setMessageVerbosity( coral::MsgLevel level ){
  coral::MessageStream::setMsgVerbosity( level );  
}


