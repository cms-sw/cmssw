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
  addOption<std::string>("read","r","read data from input file");
  addOption<bool>("generate","g","generate a new key when not specified");
  addOption<bool>("dump_template","d","dump an input file template");
}

cond::CreateKeyUtilities::~CreateKeyUtilities(){
}

int cond::CreateKeyUtilities::execute(){
  std::string inFile("");
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
  }

  if( hasOptionValue("read") ) {
    inFile = getOptionValue<std::string>("read");
    if(!inFile.empty()){
      DecodingKey key;
      key.init( inFile, Auth::COND_KEY );
      key.list( std::cout );
      return 0;
    }
    return 1;
  }

  if( hasOptionValue("dump_template") ) {
    std::cout <<DecodingKey::templateFile() <<std::endl;
    return 0;    
  }

  return 1;
}

int main( int argc, char** argv ){

  cond::CreateKeyUtilities utilities;
  return utilities.run(argc,argv);
}
