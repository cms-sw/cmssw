#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/Cipher.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cond {
  class KeyManager : public Utilities {
    public:
      KeyManager();
      ~KeyManager();
      int execute();
  };
}

cond::KeyManager::KeyManager():Utilities("cmscond_key_manager"){
  addOption<bool>("create","","[-f a0] create a credential key from input file data");
  addOption<bool>("dump","","[-p a0 -f a1 ] read the content from a key and dump it to the specified file");
  addOption<bool>("read","","[-p a0] read the content from a credential key");
  addOption<bool>("dump_template","t","(-f a0) dump an input file template (in the specified file)");
  addOption<std::string>("textFile","f","the key file in text format");
  addOption<std::string>("keyPath","p","the path of the key file");
}

cond::KeyManager::~KeyManager(){
}

int cond::KeyManager::execute(){

  if( hasOptionValue("dump_template") ) {
    if( hasOptionValue("textFile") ) {
      std::string textFile = getOptionValue<std::string>("textFile");
      std::ofstream outFile ( textFile.c_str() );
      if (outFile.is_open()){
	outFile <<DecodingKey::templateFile()<<std::endl;
	outFile.close();
      } else {
	std::cout <<"ERROR: file \""<<textFile<<"\" cannot be open."<<std::endl;
	outFile.close();
	return 1;
      }
    } else {
      std::cout <<DecodingKey::templateFile()<<std::endl;
    }
    return 0;    
  }

  if( hasOptionValue("create") ) {
    std::string textFile = getOptionValue<std::string>("textFile");
    size_t keySize = Auth::COND_AUTHENTICATION_KEY_SIZE;
    DecodingKey key;
    key.init( DecodingKey::FILE_NAME, Auth::COND_KEY, false );
    key.createFromInputFile( textFile, keySize );
    if( hasDebug() ) key.list( std::cout );
    key.flush();
    return 0;
  }

  if( !hasOptionValue("dump") && !hasOptionValue("read")) {
    return 0;
  }

  std::string path = getOptionValue<std::string>("keyPath");
  boost::filesystem::path keyPath( path );
  boost::filesystem::path keyFile( DecodingKey::FILE_PATH );
  keyPath /= keyFile;
  std::string keyFileName = keyPath.string();
  if(!boost::filesystem::exists( keyFileName )){
    std::cout <<"ERROR: specified key file \""<<keyFileName<<"\" does not exist."<<std::endl;
    return 1;
  }

  if( hasOptionValue("dump") ) {
    std::string textFile = getOptionValue<std::string>("textFile");
    std::ofstream outFile ( textFile.c_str() );
    if (outFile.is_open()){
      DecodingKey key;
      key.init( keyFileName, Auth::COND_KEY );
      key.list( outFile );
      outFile.close();
      return 0;
    } else {
      std::cout <<"ERROR: file \""<<textFile<<"\" cannot be open."<<std::endl;
      outFile.close();
      return 1;
    }
  }

  if( hasOptionValue("read") ) {
    DecodingKey key;
    key.init( keyFileName, Auth::COND_KEY );
    key.list( std::cout );
    return 0;
  }

  return 1;
}

int main( int argc, char** argv ){

  cond::KeyManager mgr;
  return mgr.run(argc,argv);
}
