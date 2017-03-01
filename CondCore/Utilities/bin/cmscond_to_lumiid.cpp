#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

namespace cond {
  class ToLumiIdUtilities : public Utilities {
    public:
      ToLumiIdUtilities();
      ~ToLumiIdUtilities();
      int execute() override;
  };
}

cond::ToLumiIdUtilities::ToLumiIdUtilities():Utilities("cmscond_to_lumiid"){
  addOption<unsigned int>("runnumber","r","run number(required)");
  addOption<unsigned int>("lumiblocknumber","l","lumi block number(required)");
}

cond::ToLumiIdUtilities::~ToLumiIdUtilities() {}

int cond::ToLumiIdUtilities::execute() {
  if( !hasOptionValue("runnumber") ){
    std::cout <<"ERROR: Missing mandatory option \"runnumber\"."<<std::endl;
    return 1;
  }
  unsigned int runnumber = getOptionValue<unsigned int>("runnumber");
  if( !hasOptionValue("lumiblocknumber") ){
    std::cout <<"ERROR: Missing mandatory option \"lumiblocknumber\"."<<std::endl;
    return 1;
  }
  unsigned int lumiblockid = getOptionValue<unsigned int>("lumiblocknumber");
  bool debug = hasDebug();

  edm::LuminosityBlockID lumiid(runnumber,lumiblockid);
 
  if(lumiid.value()<edm::LuminosityBlockID::firstValidLuminosityBlock().value()){
    std::cout<<"ERROR: Invalid Input "<<lumiid<<std::endl;
    std::cout<<"firstValidLuminosityBlockID:\t"<<edm::LuminosityBlockID::firstValidLuminosityBlock().value()<<std::endl;
    return -1;
  }
  std::cout<<lumiid<<std::endl;
  std::cout<<"LuminosityBlockID:\t"<<lumiid.value()<<std::endl;
  if(debug){
    std::cout<<"firstValidLuminosityBlockID:\t"<<edm::LuminosityBlockID::firstValidLuminosityBlock().value()<<std::endl;
    std::cout<<"maxRunNumber:\t"<<edm::RunID::maxRunNumber()<<std::endl;
    std::cout<<"maxLuminosityBlockNumber:\t"<<edm::LuminosityBlockID::maxLuminosityBlockNumber()<<std::endl;
    std::cout<<"maxLuminosityBlockID:\t"<<edm::LuminosityBlockID(edm::RunID::maxRunNumber(),edm::LuminosityBlockID::maxLuminosityBlockNumber()).value()<<std::endl;
  }
  return 0;
}

int main( int argc, char** argv ){
  cond::ToLumiIdUtilities utilities;
  return utilities.run(argc,argv);
}
