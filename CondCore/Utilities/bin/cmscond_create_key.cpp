#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <iostream>
#include "CondCore/Utilities/interface/Utilities.h"

namespace cond {
  class CreateKeyUtilities : public Utilities {
    public:
      CreateKeyUtilities();
      ~CreateKeyUtilities();
      int execute();
  };
}

cond::CreateKeyUtilities::CreateKeyUtilities():Utilities("cmscond_create_key"){
  addOption<std::string>("filename","f","encrypted filename (required)");
  addOption<std::string>("key","k","encoding key (required)");
  addOption<std::string>("password","p","password (required)");
  addOption<std::string>("outputFile","o","output key file (required)");
}

cond::CreateKeyUtilities::~CreateKeyUtilities() {}

int cond::CreateKeyUtilities::execute() {
  if( !hasOptionValue("filename") ){
    std::cout <<"ERROR: Missing mandatory option \"filename\"."<<std::endl;
    return 1;
  }
  std::string fileName = getOptionValue<std::string>("filename");
  if( !hasOptionValue("key") ){
    std::cout <<"ERROR: Missing mandatory option \"key\"."<<std::endl;
    return 1;
  }
  std::string key = getOptionValue<std::string>("key");
  if( !hasOptionValue("password") ){
    std::cout <<"ERROR: Missing mandatory option \"password\"."<<std::endl;
    return 1;
  }
  std::string password = getOptionValue<std::string>("password");
  if( !hasOptionValue("outputFile") ){
    std::cout <<"ERROR: Missing mandatory option \"outputFile\"."<<std::endl;
    return 1;
  }
  std::string outputFile = getOptionValue<std::string>("outputFile");
  cond::DecodingKey::createFile(password,key,fileName,outputFile);
  return 0;
}

int main( int argc, char** argv ){

  cond::CreateKeyUtilities utilities;
  return utilities.run(argc,argv);
}
