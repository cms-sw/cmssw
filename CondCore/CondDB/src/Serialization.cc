#include "CondCore/CondDB/interface/Serialization.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
//
#include <sstream>
// root includes 
#include "TBufferFile.h"
#include "TClass.h"
#include "Cintex/Cintex.h"

namespace cond {

  struct CintexIntializer {
    static bool init;
    CintexIntializer(){
      if (!init) {
	ROOT::Cintex::Cintex::Enable();
	init = true;
      }      
    }
  };

  bool CintexIntializer::init = false;

  // initialize Cintex and load dictionary when required
  TClass* lookUpDictionary( const std::type_info& sourceType ){
    static CintexIntializer initializer;
    TClass* rc = TClass::GetClass(sourceType);
    if( !rc ){
      static std::string const prefix("LCGReflex/");
      std::string name = demangledName(sourceType);
      edmplugin::PluginCapabilities::get()->load(prefix + name);
      rc = TClass::GetClass(sourceType);
    }
    return rc;
  }
}

cond::RootOutputArchive::RootOutputArchive( std::ostream& dest ):
  m_buffer( dest ){
} 

void cond::RootOutputArchive::write( const std::type_info& sourceType, const void* sourceInstance){
  TClass* r_class = lookUpDictionary( sourceType );
  if (!r_class) throwException( "No ROOT class registered for \"" + demangledName(sourceType)+"\"", "RootOutputArchive::write");
  TBufferFile buffer(TBufferFile::kWrite);
  buffer.InitMap();
  buffer.StreamObject(const_cast<void*>(sourceInstance), r_class);
  // copy the stream into the target buffer
  m_buffer.write( static_cast<const char*>(buffer.Buffer()), buffer.Length() ); 
}

cond::RootInputArchive::RootInputArchive( std::istream& source ):
  m_buffer( std::istreambuf_iterator<char>(source), std::istreambuf_iterator<char>()),
  m_streamer( new TBufferFile( TBufferFile::kRead, m_buffer.size(), const_cast<char*>(m_buffer.c_str()), kFALSE ) ){
  m_streamer->InitMap();
}

cond::RootInputArchive::~RootInputArchive(){
  delete m_streamer;
}

void cond::RootInputArchive::read( const std::type_info& destinationType, void* destinationInstance){
  TClass* r_class = lookUpDictionary( destinationType );
  if (!r_class) throwException( "No ROOT class registered for \"" + demangledName(destinationType) +"\"","RootInputArchive::read");
  m_streamer->StreamObject(destinationInstance, r_class);   
}

