#include "FWCore/PluginManager/interface/PluginManager.h"
#include "FWCore/PluginManager/interface/standard.h"
#include "FWCore/Utilities/interface/TypeWithDict.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "MyTestData.h"
//
#include <iostream>
#include <sstream>
#include "TClass.h"
#include "TBufferFile.h"

int main (int argc, char** argv)
{
  
  int ret = 0;
  edmplugin::PluginManager::Config config;
  edmplugin::PluginManager::configure(edmplugin::standard::config());

  if (!edm::TypeWithDict::byName("MyTestData")) {
    throw cms::Exception("DictionaryMissingClass") << "The dictionary of class 'MyTestData' is missing!";
  }

  MyTestData d0( 17 );
  d0.print();
  TClass* r_class = TClass::GetClass(typeid(MyTestData));
  if (!r_class) throw std::string( "No ROOT class registered for MyTestData");
  TBufferFile wstreamer(TBufferFile::kWrite);
  wstreamer.InitMap();
  //wstreamer.StreamObject(&d0, r_class);
  wstreamer.WriteObjectAny(&d0, r_class);

  std::ostringstream wbuff;
  wbuff.write( static_cast<const char*>(wstreamer.Buffer()), wstreamer.Length() ); 
  //
  std::string rbuff = wbuff.str();
  TBufferFile rstreamer(  TBufferFile::kRead, rbuff.size(), const_cast<char*>(rbuff.c_str()),  kFALSE );  
  rstreamer.InitMap();
  //MyTestData d1;
  //rstreamer.StreamObject(&d1, r_class);   
  void* obj = rstreamer.ReadObjectAny( r_class );   
  MyTestData& d1 = *(static_cast<MyTestData*>(obj));
  d1.print();

  return ret;
}

