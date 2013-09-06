#include "CondCore/CondDB/interface/Streamer.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
//
#include <sstream>
// boost includes
#include <boost/iostreams/filtering_streambuf.hpp>
//#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/gzip.hpp>
// root includes 
#include "TBufferFile.h"
#include "TClass.h"
#include "Cintex/Cintex.h"

namespace conddb {

  // initialize Cintex and load dictionary when required
  TClass* lookUpDictionary( const std::type_info& sourceType ){
    static bool cintexInitialized = false;
    if (!cintexInitialized) {
      ROOT::Cintex::Cintex::Enable();
      cintexInitialized = true;
    }
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

conddb::RootOutputArchive::RootOutputArchive( boost::iostreams::filtering_streambuf<boost::iostreams::output>& inputData ):
  m_buffer( inputData ){
} 

void conddb::RootOutputArchive::write( const std::type_info& sourceType, const void* sourceInstance){
  TClass* r_class = lookUpDictionary( sourceType );
  if (!r_class) throwException( "No ROOT class registered for \"" + demangledName(sourceType)+"\"", "RootOutputArchive::write");
  TBufferFile buffer(TBufferFile::kWrite);
  buffer.InitMap();
  buffer.StreamObject(const_cast<void*>(sourceInstance), r_class);
  // copy the stream into the target buffer
  m_buffer.sputn( static_cast<const char*>(buffer.Buffer()), buffer.Length() ); 
}

conddb::RootInputArchive::RootInputArchive( boost::iostreams::filtering_streambuf<boost::iostreams::input>& inputData ){
  // many copies... we shpould do better than this...
  std::stringstream tmp;
  boost::iostreams::copy( inputData, tmp );
  m_copy = tmp.str();
  m_buffer =  new TBufferFile( TBufferFile::kRead, m_copy.size(), const_cast<char*>(m_copy.c_str()), kFALSE );
  m_buffer->InitMap();
}

conddb::RootInputArchive::~RootInputArchive(){
  delete m_buffer;
}

void conddb::RootInputArchive::read( const std::type_info& destinationType, void* destinationInstance){
  TClass* r_class = lookUpDictionary( destinationType );
  if (!r_class) throwException( "No ROOT class registered for \"" + demangledName(destinationType) +"\"","RootInputArchive::read");
  m_buffer->StreamObject(destinationInstance, r_class);   
}

conddb::OutputStreamer::OutputStreamer():
  m_data(){
}

const conddb::Binary& conddb::OutputStreamer::data() const{
  return m_data;
}

conddb::InputStreamer::InputStreamer( const std::string& payloadType, const Binary& payloadData ):
  m_objectType( payloadType ),
  m_outBuf(){
  m_outBuf.write( static_cast<const char*>(payloadData.data()), payloadData.size() );
}

