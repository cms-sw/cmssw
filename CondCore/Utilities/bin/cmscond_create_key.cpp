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
<<<<<<< cmscond_create_key.cpp
=======
  addOption<size_t>("generate","g","generate a new key for every service");
>>>>>>> 1.5
  addOption<std::string>("read","r","read data from input file");
<<<<<<< cmscond_create_key.cpp
  addOption<bool>("generate","g","generate a new key when not specified");
  addOption<bool>("dump_template","d","dump an input file template");
=======
  addOption<std::string>("clone","k","clone from the input key");
>>>>>>> 1.5
}

cond::CreateKeyUtilities::~CreateKeyUtilities(){
}

int cond::CreateKeyUtilities::execute(){
  std::string inFile("");
<<<<<<< cmscond_create_key.cpp
  if( hasOptionValue("create") ) {
    inFile = getOptionValue<std::string>("create");
    size_t keySize = 0;
    if( hasOptionValue("generate") ) keySize = Auth::COND_AUTHENTICATION_KEY_SIZE;
    if(!inFile.empty()){
      DecodingKey key;
      key.init( DecodingKey::FILE_NAME, Auth::COND_KEY, false );
      key.createFromInputFile( inFile, keySize );
      if( hasDebug() ) key.list( std::cout );
      key.flush();
      return 0;
    }
    return 1;
=======
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
>>>>>>> 1.5
  }

  if( hasOptionValue("read") ) {
    inFile = getOptionValue<std::string>("read");
    if(!inFile.empty()){
      DecodingKey key;
      key.init( DecodingKey::FILE_NAME, Auth::COND_KEY );
      key.list( std::cout );
      return 0;
    }
    return 1;
  }

<<<<<<< cmscond_create_key.cpp
  if( hasOptionValue("dump_template") ) {
    std::cout <<DecodingKey::templateFile() <<std::endl;
    return 0;    
=======
  if( hasOptionValue("clone") ) {
    std::string inputKeyFile = getOptionValue<std::string>("clone");
    DecodingKey inputKey;
    inputKey.init( inputKeyFile, Auth::COND_KEY );
    
>>>>>>> 1.5
  }

  return 1;
}

int main( int argc, char** argv ){

  cond::CreateKeyUtilities utilities;
  return utilities.run(argc,argv);
}
