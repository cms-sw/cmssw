#include "CondCore/ORA/interface/Configuration.h"
#include "CondCore/ORA/interface/IBlobStreamingService.h"
#include "CondCore/ORA/interface/IReferenceHandler.h"
// 
#include <cstdlib>
#include <string.h>

static const char* CORAL_MSG_LEVEL = "CORAL_MSG_LEVEL";
coral::MsgLevel coralMessageLevel( const char* envVar ){
  coral::MsgLevel ret = coral::Nil;
  if( ::strcmp(envVar,"VERBOSE")==0 || ::strcmp(envVar,"Verbose")==0 ) ret = coral::Verbose;
  if( ::strcmp(envVar,"DEBUG")==0 || ::strcmp(envVar,"Debug")==0 ) ret = coral::Debug;
  if( ::strcmp(envVar,"INFO")==0 || ::strcmp(envVar,"Info")==0 ) ret = coral::Info;
  if( ::strcmp(envVar,"WARNING")==0 || ::strcmp(envVar,"Warning")==0 ) ret = coral::Warning;
  if( ::strcmp(envVar,"ERROR")==0 || ::strcmp(envVar,"Error")==0 ) ret = coral::Error;
  if( ::strcmp(envVar,"FATAL")==0 || ::strcmp(envVar,"Fatal")==0 ) ret = coral::Fatal;
  if( ::strcmp(envVar,"ALWAYS")==0 || ::strcmp(envVar,"Always")==0 ) ret = coral::Always;
  if( ::strcmp(envVar,"NUMLEVELS")==0 || ::strcmp(envVar,"NumLevels")==0 ) ret = coral::NumLevels;
  return ret;
}

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
  
  const char* envVar = ::getenv( CORAL_MSG_LEVEL );
  if( envVar ){
    coral::MsgLevel level = coralMessageLevel( envVar );
    if( level != coral::Nil ) coral::MessageStream::setMsgVerbosity( coralMessageLevel( envVar ) );
  }
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


