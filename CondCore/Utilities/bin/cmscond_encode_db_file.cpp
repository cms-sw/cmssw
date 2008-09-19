#include "CoralCommon/Cipher.h"
#include "CondCore/DBCommon/interface/FileUtils.h"
#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <boost/program_options.hpp>
#include <fstream>
#include <iostream>


int main( int argc, char** argv ){

  boost::program_options::options_description desc("options");
  boost::program_options::options_description visible("Usage: cmscond_encode_db_file [options] \n");
  visible.add_options()
    ("inputFileName,i",boost::program_options::value<std::string>(),"input filename (optional, def=authentication.xml)")
    ("outputFileName,o",boost::program_options::value<std::string>(),"output filename (optional, def=database.dat)")
    ("encodingKey,k",boost::program_options::value<std::string>(),"encoding password (required)")
    ("decode,d","decode (optional)")
    ("debug","switch on debug mode")
    ("help,h", "help message")
    ;
  desc.add(visible);
  std::string inputFileName("authentication.xml");
  std::string outputFileName("database.dat");
  std::string key("");
  bool decode=false;
  bool debug=false;

  boost::program_options::variables_map vm;
  try{
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).run(), vm);
    if (vm.count("help")) {
      std::cout << visible <<std::endl;;
      return 0;
    }
    if( vm.count("inputFileName") ){
      inputFileName=vm["inputFileName"].as<std::string>();
    }
    if( vm.count("outputFileName")){
      outputFileName=vm["outputFileName"].as<std::string>();
    }
    if(vm.count("encodingKey")){
      key=vm["encodingKey"].as<std::string>();
    }
    if(key.empty()){
      std::cerr <<"[Error] no encodingKey[k] option given \n";
      std::cerr<<" please do "<<argv[0]<<" --help \n";
      return 1;
    }
    if(vm.count("decode")){
      decode=true;
    }
    if(vm.count("debug")){
      debug=true;
    }
    boost::program_options::notify(vm);
  }catch(const boost::program_options::error& er) {
    std::cerr << er.what()<<std::endl;
    return 1;
  }

  cond::FileReader inputFile;
  std::string cont("");
  try{
    inputFile.read(inputFileName);
    cont = inputFile.content();
    cond::DecodingKey::validatePassword(key);
  } catch (const cond::Exception& exc){
    std::cerr << exc.what()<<std::endl;
    return 1;
  }

  std::string outputData =(decode? coral::Cipher::decode(cont,key): coral::Cipher::encode(cont,key));
  if(debug){
    std::cout << "inputFileName=\""<<inputFileName<<"\""<<std::endl;
    std::cout << "outputFileName=\""<<outputFileName<<"\""<<std::endl;
    std::cout << "encodingKey=\""<<key<<"\""<<std::endl;
    std::cout << "decoding="<<(decode?std::string("true"):std::string("false"))<<std::endl;
    std::cout << "Output file content:"<<std::endl;
    std::cout << outputData << std::endl;
  }
  
  std::ofstream outputFile;
  outputFile.open(outputFileName.c_str());
  if(!outputFile.good()){
    std::cerr << "Cannot open the output file \""<<outputFileName<<"\""<<std::endl;
    outputFile.close();
    return 1;
  }
  outputFile << outputData;
  outputFile.flush();
  outputFile.close();
  std::cout << "File \""<< inputFileName<< "\" content encoded in file \"" << outputFileName << "\""<< std::endl;
  
  // test !!!
  if(debug && !decode){
    cond::FileReader filer;
    std::string outCont("");
    try{
      filer.read(outputFileName);
      outCont = filer.content();
    } catch (const cond::Exception& exc){
      std::cerr << exc.what()<<std::endl;
      return 1;
    }
    std::string decodedData = coral::Cipher::decode(outCont,key);
    std::cout << "Decoded output file content:"<<std::endl;
    std::cout << decodedData << std::endl;
  }
  
  
  return 0;
}
