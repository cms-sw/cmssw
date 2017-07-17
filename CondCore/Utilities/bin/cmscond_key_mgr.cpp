#include "CondCore/Utilities/interface/Utilities.h"
#include "CondCore/CondDB/interface/DecodingKey.h"
#include "CondCore/CondDB/interface/Cipher.h"
#include "CondCore/CondDB/interface/Auth.h"
#include "CondCore/CondDB/interface/Exception.h"

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace cond {
  class KeyMgrUtilities : public Utilities {
    public:
      KeyMgrUtilities();
      ~KeyMgrUtilities();
      int execute() override;
  };
}

cond::KeyMgrUtilities::KeyMgrUtilities():Utilities("cmscond_key_mgr"){
  addOption<std::string>("create","c","creating from input file data");
  addOption<std::string>("read","r","read data from input file");
  addOption<bool>("generate","g","generate a new key when not specified");
  addOption<bool>("dump_template","d","dump an input file template");
}

cond::KeyMgrUtilities::~KeyMgrUtilities(){
}

int cond::KeyMgrUtilities::execute(){
  std::string inFile("");
  if( hasOptionValue("create") ) {
    inFile = getOptionValue<std::string>("create");
    size_t keySize = 0;
    if( hasOptionValue("generate") ) keySize = auth::COND_AUTHENTICATION_KEY_SIZE;
    if(!inFile.empty()){
      auth::DecodingKey key;
      key.init( auth::DecodingKey::FILE_NAME, auth::COND_KEY, false );
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
      auth::DecodingKey key;
      key.init( inFile, auth::COND_KEY );
      key.list( std::cout );
      return 0;
    }
    return 1;
  }

  if( hasOptionValue("dump_template") ) {
    std::cout <<auth::DecodingKey::templateFile() <<std::endl;
    return 0;    
  }

  return 1;
}

int main( int argc, char** argv ){

  cond::KeyMgrUtilities utilities;
  return utilities.run(argc,argv);
}
