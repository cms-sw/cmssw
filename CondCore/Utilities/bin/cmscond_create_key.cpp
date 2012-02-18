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
  addOption<bool>("generate","g","generate a new key for every service");
  addOption<std::string>("read","r","read data from input file");
  addOption<std::string>("encode","e","only encode the input data");
  addOption<std::string>("decode","d","only decode the input data");
  addOption<std::string>("key","k","the encoding key");
  addOption<std::string>("file","f","the input/output file");
}

cond::CreateKeyUtilities::~CreateKeyUtilities(){
}

int cond::CreateKeyUtilities::execute(){
  std::string inFile("");
  if( hasOptionValue("create") ) inFile = getOptionValue<std::string>("create");
  bool generate = hasOptionValue("generate");
  
  if(!inFile.empty()){
    DecodingKey key;
    key.init( DecodingKey::FILE_NAME, Auth::COND_KEY, false );
    key.createFromInputFile( inFile, generate );
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

  std::string input("");
  std::string key("");
  if( hasOptionValue("key") ) key = getOptionValue<std::string>("key");
  std::string fileOut("");
  bool outFile = hasOptionValue("file");
  if( outFile ) fileOut = getOptionValue<std::string>("file");
  bool encode = hasOptionValue("encode");
  bool decode = hasOptionValue("decode");
  if( encode ){
    input = getOptionValue<std::string>("encode");
    Cipher cipher( key );
    if( outFile ) {
      std::ofstream fout(fileOut);
      cipher.bencrypt( input, fout );
      fout.close();
    } else {
      std::cout<< cipher.encrypt( input )<<std::endl;
    }
    return 0; 
  }
  if( decode ){
    input = getOptionValue<std::string>("decode");
    Cipher cipher( key );
    std::string decoded("");
    if( outFile ) {
      std::ifstream fin( fileOut );
      decoded = cipher.bdecrypt( fin );
      fin.close();
      std::cout <<"Done. s="<<decoded<<std::endl;
    } else {
      decoded = cipher.decrypt( input );
    }
    std::cout<< decoded<<std::endl;
    return 0; 
  }
  
  
  return 1;
}

int main( int argc, char** argv ){

  cond::CreateKeyUtilities utilities;
  return utilities.run(argc,argv);
}
