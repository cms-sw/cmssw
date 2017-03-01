#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/CondDB/interface/Exception.h"

#include "CondCore/Utilities/interface/Utilities.h"
#include <iostream>

namespace cond {
  class FromLumiIdUtilities : public Utilities {
    public:
      FromLumiIdUtilities();
      ~FromLumiIdUtilities();
      int execute() override;
  };
}

cond::FromLumiIdUtilities::FromLumiIdUtilities():Utilities("cmscond_from_lumiid"){
  addOption<boost::uint64_t>("lumiid","i","luminosity block id of unsigned 64bit int(required)");
}

cond::FromLumiIdUtilities::~FromLumiIdUtilities() {}

int cond::FromLumiIdUtilities::execute() {
  if( !hasOptionValue("lumiid") ){
    std::cout <<"ERROR: Missing mandatory option \"lumiid\"."<<std::endl;
    return 1;
  }
  boost::uint64_t lumiid = getOptionValue<boost::uint64_t>("lumiid");
  bool debug = hasDebug();
  std::cout<<edm::LuminosityBlockID(lumiid)<<std::endl;
  if(debug){
    std::cout<<"firstValidLuminosityBlockID:\t"<<edm::LuminosityBlockID::firstValidLuminosityBlock().value()<<std::endl;
    std::cout<<"maxRunNumber:\t"<<edm::RunID::maxRunNumber()<<std::endl;
    std::cout<<"maxLuminosityBlockNumber:\t"<<edm::LuminosityBlockID::maxLuminosityBlockNumber()<<std::endl;
    std::cout<<"maxLuminosityBlockID:\t"<<edm::LuminosityBlockID(edm::RunID::maxRunNumber(),edm::LuminosityBlockID::maxLuminosityBlockNumber()).value()<<std::endl;
  }
  return 0;
}

int main( int argc, char** argv ){
  cond::FromLumiIdUtilities utilities;
  return utilities.run(argc,argv);
}
