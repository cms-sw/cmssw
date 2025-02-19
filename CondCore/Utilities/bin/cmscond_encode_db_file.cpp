#include "CoralCommon/Cipher.h"
#include "CondCore/DBCommon/interface/FileUtils.h"
#include "CondCore/DBCommon/interface/DecodingKey.h"
#include "CondCore/DBCommon/interface/Exception.h"

#include <fstream>
#include <iostream>
#include "CondCore/Utilities/interface/Utilities.h"

namespace cond {
  class EncodeDbFileUtilities : public Utilities {
    public:
      EncodeDbFileUtilities();
      ~EncodeDbFileUtilities();
      int execute();
  };
}

cond::EncodeDbFileUtilities::EncodeDbFileUtilities():Utilities("cmscond_encode_db_file"){
  addOption<std::string>("inputFileName","i","input filename (optional, def=authentication.xml)");
  addOption<std::string>("outputFileName","o","output filename (required)");
  addOption<std::string>("encodingKey","k","encoding password (required)");
  addOption<bool>("decode","d","decode (optional)");
}

cond::EncodeDbFileUtilities::~EncodeDbFileUtilities() {}

int cond::EncodeDbFileUtilities::execute() {
  std::string inputFileName("authentication.xml");
  if( hasOptionValue("inputFileName") ){
    inputFileName = getOptionValue<std::string>("inputFileName");
  }
  if( !hasOptionValue("outputFileName") ){
    std::cout <<"ERROR: Missing mandatory option \"outputFileName\"."<<std::endl;
    return 1;
  }
  std::string outputFileName = getOptionValue<std::string>("outputFileName");
  if( !hasOptionValue("encodingKey") ){
    std::cout <<"ERROR: Missing mandatory option \"encodingKey\"."<<std::endl;
    return 1;
  }
  std::string key = getOptionValue<std::string>("encodingKey");
  bool decode = hasOptionValue("decode");
  bool debug = hasDebug();

  cond::FileReader inputFile;
  std::string cont("");

  inputFile.read(inputFileName);
  cont = inputFile.content();
  //cond::DecodingKey::validateKey(key);
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
    filer.read(outputFileName);
    outCont = filer.content();
    std::string decodedData = coral::Cipher::decode(outCont,key);
    std::cout << "Decoded output file content:"<<std::endl;
    std::cout << decodedData << std::endl;
  }

  return 0;
}

int main( int argc, char** argv ){

  cond::EncodeDbFileUtilities utilities;
  return utilities.run(argc,argv);
}
