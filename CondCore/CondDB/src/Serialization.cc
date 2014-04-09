#include "CondCore/CondDB/interface/Serialization.h"
#include "CondCore/CondDB/interface/Exception.h"
#include "CondCore/CondDB/interface/Utils.h"
#include "FWCore/PluginManager/interface/PluginCapabilities.h"
//
#include <sstream>
// root includes 
#include "TStreamerInfo.h"
#include "TClass.h"
#include "TList.h"
#include "TBufferFile.h"
#include "Cintex/Cintex.h"

namespace cond {

  struct CintexIntializer {
    CintexIntializer(){
      ROOT::Cintex::Cintex::Enable();
    }
  };

  // initialize Cintex and load dictionary when required
  TClass* lookUpDictionary( const std::type_info& sourceType ){
    static const CintexIntializer initializer;
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

class RootStreamBuffer: public TBufferFile {
public:
  RootStreamBuffer():
    TBufferFile( TBufferFile::kWrite ),
    m_streamerInfoBuff( TBufferFile::kWrite ){
  }

  RootStreamBuffer( const std::string& dataSource, const std::string& streamerInfoSource ):
    TBufferFile( TBufferFile::kRead, dataSource.size(), const_cast<char*>( dataSource.c_str()), kFALSE ),
    m_streamerInfoBuff( TBufferFile::kRead, streamerInfoSource.size(), const_cast<char*>( streamerInfoSource.c_str()), kFALSE ){
  }

  void ForceWriteInfo(TVirtualStreamerInfo* sinfo, Bool_t /* force */){
    m_streamerInfo.Add( sinfo );
  }

  void TagStreamerInfo(TVirtualStreamerInfo* sinfo){
    m_streamerInfo.Add( sinfo );
  }

  void write( const void* obj, const TClass* ptrClass ){
    m_streamerInfo.Clear();
    // this will populate the streamerInfo list 'behind the scenes' - calling the TagStreamerInfo method
    StreamObject(const_cast<void*>(obj), ptrClass);    
    // serialize the StreamerInfo
    if(m_streamerInfo.GetEntries() ){
      m_streamerInfoBuff.WriteObject( &m_streamerInfo );
    }
    m_streamerInfo.Clear();
  }

  void read( void* destinationInstance, const TClass* ptrClass ){
    // first "load" the available streaminfo(s) 
    // code imported from TSocket::RecvStreamerInfos
    TList *list = 0;
    if(m_streamerInfoBuff.Length()){
      list = (TList*)m_streamerInfoBuff.ReadObject( TList::Class() );
      TIter next(list);
      TStreamerInfo *info;
      TObjLink *lnk = list->FirstLink();
      // First call BuildCheck for regular class
      while (lnk) {
	info = (TStreamerInfo*)lnk->GetObject();
	TObject *element = info->GetElements()->UncheckedAt(0);
	Bool_t isstl = element && strcmp("This",element->GetName())==0;
	if (!isstl) {
	  info->BuildCheck();
	}
	lnk = lnk->Next();
      }
      // Then call BuildCheck for stl class
      lnk = list->FirstLink();
      while (lnk) {
	info = (TStreamerInfo*)lnk->GetObject();
	TObject *element = info->GetElements()->UncheckedAt(0);
	Bool_t isstl = element && strcmp("This",element->GetName())==0;
	if (isstl) {
	  info->BuildCheck();
	}
	lnk = lnk->Next();
      }
    }
    // then read the object data
    StreamObject(destinationInstance, ptrClass);
    if( list ) delete list;
  }

  void copy( std::ostream& destForData, std::ostream& destForStreamerInfo ){
    destForData.write( static_cast<const char*>(Buffer()),Length() );
    destForStreamerInfo.write( static_cast<const char*>(m_streamerInfoBuff.Buffer()),m_streamerInfoBuff.Length() );
  }
  
private:
  TBufferFile m_streamerInfoBuff;
  TList m_streamerInfo;
};

cond::RootOutputArchive::RootOutputArchive( std::ostream& dataDest, std::ostream& streamerInfoDest ):
  m_dataBuffer( dataDest ),
  m_streamerInfoBuffer( streamerInfoDest ){
} 

void cond::RootOutputArchive::write( const std::type_info& sourceType, const void* sourceInstance){
  TClass* r_class = lookUpDictionary( sourceType );
  if (!r_class) throwException( "No ROOT class registered for \"" + demangledName(sourceType)+"\"", "RootOutputArchive::write");
  RootStreamBuffer buffer;
  buffer.InitMap();
  buffer.write(sourceInstance, r_class);
  // copy the two streams into the target buffers
  buffer.copy( m_dataBuffer, m_streamerInfoBuffer );
}

cond::RootInputArchive::RootInputArchive( std::istream& binaryData, std::istream& binaryStreamerInfo ):
  m_dataBuffer( std::istreambuf_iterator<char>( binaryData ), std::istreambuf_iterator<char>()),
  m_streamerInfoBuffer( std::istreambuf_iterator<char>( binaryStreamerInfo ), std::istreambuf_iterator<char>()),
  m_streamer( new RootStreamBuffer( m_dataBuffer, m_streamerInfoBuffer ) ){
  m_streamer->InitMap();
}

cond::RootInputArchive::~RootInputArchive(){
  delete m_streamer;
}

void cond::RootInputArchive::read( const std::type_info& destinationType, void* destinationInstance){
  TClass* r_class = lookUpDictionary( destinationType );
  if (!r_class) throwException( "No ROOT class registered for \"" + demangledName(destinationType) +"\"","RootInputArchive::read");
  m_streamer->read( destinationInstance, r_class );
}

