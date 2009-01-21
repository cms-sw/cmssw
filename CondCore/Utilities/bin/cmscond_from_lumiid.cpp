#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <boost/program_options.hpp>
#include <iostream>

int main( int argc, char** argv ){
  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_from_lumiid [options] \n");
  visible.add_options()
    ("lumiid,i",boost::program_options::value<boost::uint64_t>(),"luminosity block id of unsigned 64bit int(required)")
    ("debug","switch on debug mode")
    ("help,h", "help message")
    ;
  desc.add(visible);
  bool debug=false;
  boost::uint64_t lumiid;
  boost::program_options::variables_map vm;
  try{
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help")) {
      std::cout << visible <<std::endl;;
      return 0;
    }
    if( vm.count("lumiid") ){
      lumiid=vm["lumiid"].as<boost::uint64_t>();
    }else{
      std::cout<<"option --lumiid or -i is required"<<std::endl;
      return -1;
    }
    if(vm.count("debug")){
      debug=true;
    }
    boost::program_options::notify(vm);
  }catch(const boost::program_options::error& er) {
    std::cerr << er.what()<<std::endl;
    return 1;
  }
  std::cout<<edm::LuminosityBlockID(lumiid)<<std::endl;
  if(debug){
    std::cout<<"firstValidLuminosityBlockID:\t"<<edm::LuminosityBlockID::firstValidLuminosityBlock().value()<<std::endl;
    std::cout<<"maxRunNumber:\t"<<edm::RunID::maxRunNumber()<<std::endl;
    std::cout<<"maxLuminosityBlockNumber:\t"<<edm::LuminosityBlockID::maxLuminosityBlockNumber()<<std::endl;
    std::cout<<"maxLuminosityBlockID:\t"<<edm::LuminosityBlockID(edm::RunID::maxRunNumber(),edm::LuminosityBlockID::maxLuminosityBlockNumber()).value()<<std::endl;
  }
  return 0;
}
