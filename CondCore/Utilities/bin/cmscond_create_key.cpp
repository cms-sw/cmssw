#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/Cipher.h"
#include "CondCore/DBCommon/interface/Auth.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cond {
  class CreateKeyUtilities : public Utilities {
    public:
      CreateKeyUtilities();
      ~CreateKeyUtilities();
      int execute();
  };
}

cond::CreateKeyUtilities::CreateKeyUtilities():Utilities("cmscond_create_key"){
  addOption<std::string>("create","c","creating from input file data");
  addOption<size_t>("generate","g","generate a new key for every service");
  addOption<std::string>("read","r","read data from input file");
  addOption<std::string>("clone","k","clone from the input key");
}

cond::CreateKeyUtilities::~CreateKeyUtilities(){
}

int cond::CreateKeyUtilities::execute(){
  std::string inFile("");
  if( hasOptionValue("create") ) inFile = getOptionValue<std::string>("create");
  size_t keySize = 0;
  if( hasOptionValue("generate") ) keySize = getOptionValue<size_t>("generate");
  
  if(!inFile.empty()){
    DecodingKey key;
    key.init( DecodingKey::FILE_NAME, Auth::COND_KEY, false );
    key.createFromInputFile( inFile, keySize );
    if( hasDebug() ) key.list( std::cout );
    key.flush();
    return 0;
  }

  if( hasOptionValue("read") ) inFile = getOptionValue<std::string>("read");
  if(!inFile.empty()){
    DecodingKey key;
    key.init( DecodingKey::FILE_NAME, Auth::COND_KEY );
    if( hasDebug() ) key.list( std::cout );
    return 0;
  }

  if( hasOptionValue("clone") ) {
    std::string inputKeyFile = getOptionValue<std::string>("clone");
    DecodingKey inputKey;
    inputKey.init( inputKeyFile, Auth::COND_KEY );
    
  }

  return 1;
}

int main( int argc, char** argv ){

  cond::CreateKeyUtilities utilities;
  return utilities.run(argc,argv);
}
