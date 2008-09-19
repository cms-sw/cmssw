#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <stdexcept>

int main( int argc, char** argv ){

  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_create_key_file [options] \n");
  visible.add_options()
    ("fileName,f",boost::program_options::value<std::string>(),"encrypted filename (required)")
    ("password,p",boost::program_options::value<std::string>(),"encoding password (required)")
    ("outputFile,o",boost::program_options::value<std::string>(),"output key file (required)")
    ("debug","switch on debug mode")
    ("help,h", "help message")
    ;
  desc.add(visible);
  std::string fileName("");
  std::string password("");
  std::string outputFile("");
  bool debug=false;

  boost::program_options::variables_map vm;
  try{
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help")) {
      std::cout << visible <<std::endl;;
      return 0;
    }
    if( !vm.count("fileName") ){
      std::cerr <<"[Error] no fileName[f] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
    } else {
      fileName=vm["fileName"].as<std::string>();
    }
    if( !vm.count("password") ){
      std::cerr <<"[Error] no password[p] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
    } else {
      password=vm["password"].as<std::string>();
    }
    if( !vm.count("outputFile") ){
      std::cerr <<"[Error] no outputFile[o] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
    } else {
      outputFile=vm["outputFile"].as<std::string>();
    }
    if(vm.count("debug")){
      debug=true;
    }
    boost::program_options::notify(vm);
  } catch(const boost::program_options::error& er) {
    std::cerr << er.what()<<std::endl;
    return 1;
  }

  try{
    cond::DecodingKey::createFile(password,fileName,outputFile);
  } catch (const cond::Exception& ex){
    std::cout<<"error "<<ex.what()<<std::endl;
    return 1;
  }catch(const std::exception& er){
    std::cout<<"std error "<<er.what()<<std::endl;
    return 1;
  }
  
  return 0;
}
