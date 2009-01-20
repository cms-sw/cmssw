#include "DataFormats/Provenance/interface/LuminosityBlockID.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <boost/program_options.hpp>
#include <iostream>

int main( int argc, char** argv ){
  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_to_lumiid [options] \n");
  visible.add_options()
    ("runnumber,r",boost::program_options::value<unsigned int>(),"run number(required)")
    ("lumiblocknumber,l",boost::program_options::value<unsigned int>(),"lumi block number(required)")
    ("debug","switch on debug mode")
    ("help,h", "help message")
    ;
  desc.add(visible);
  bool debug=false;
  unsigned int runnumber;
  unsigned int lumiblockid;

  boost::program_options::variables_map vm;
  try{
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help")) {
      std::cout << visible <<std::endl;;
      return 0;
    }
    if( vm.count("runnumber") ){
      runnumber=vm["runnumber"].as<unsigned int>();
    }else{
      std::cout<<"option --runnumber or -r is required"<<std::endl;
      return -1;
    }
    if( vm.count("lumiblocknumber")){
      lumiblockid=vm["lumiblocknumber"].as<unsigned int>();
    }else{
      std::cout<<"option --lumiblocknumber or -l is required"<<std::endl;
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
    std::cout<<"maxLuminosityBlockNumber:\t"<<edm::LuminosityBlockID::maxLuminosityBlockNumber()<<std::endl;
  }
  return 0;
}
