#include "RecoEgamma/EgammaTools/interface/GBRForestTools.h"

#include <iostream>
#include <fstream>

namespace {

  // Will return position of n-th occurence of a char in a string.
  // From https://stackoverflow.com/questions/18972258/index-of-nth-occurrence-of-the-string
  int strpos(std::string haystack, char needle, unsigned int nth)
  {
          std::string read;    // A string that will contain the read part of the haystack
          for (unsigned int i=1 ; i<nth+1 ; ++i)
          {
                  std::size_t found = haystack.find(needle);
                  read += haystack.substr(0,found+1); // the read part of the haystack is stocked in the read string
                  haystack.erase(0, found+1);     // remove the read part of the haystack up to the i-th needle
                  if (i == nth)
                  {
                          return read.size();
                  }
          }
          return -1;
  }

  // To get the substring between the n1th and n2th quotation mark in a string
  std::string get_substring(std::string str, int n1, int n2)
  {
      int pos = strpos(str, '"', n1);
      int count = strpos(str, '"', n2) - pos;
      return str.substr(pos, count - 1);
  }

};

std::unique_ptr<const GBRForest> GBRForestTools::createGBRForest(const std::string &weightFile){
  edm::FileInPath weightFileEdm(weightFile);
  return GBRForestTools::createGBRForest(weightFileEdm);
}

// Creates a pointer to new GBRForest corresponding to a TMVA weights file
std::unique_ptr<const GBRForest> GBRForestTools::createGBRForest(const edm::FileInPath &weightFile){

  std::string method;

  unsigned int NVar = 0;
  unsigned int NSpec = 0;

  std::vector<float> dumbVars;
  std::vector<float> dumbSpecs;

  std::vector<std::string> varNames;
  std::vector<std::string> specNames;

  std::string line;
  std::ifstream f (weightFile.fullPath());

  while(std::getline(f, line)) {
      // Terminate reading of weights file
      if (line.find("<Classes ") != std::string::npos) break;

      // Method name
      else if (line.find("<MethodSetup Method=") != std::string::npos) {
          method = get_substring(line, 1, 2);
      }

      // Number of variables
      else if (line.find("<Variables NVar=") != std::string::npos) {
          NVar = std::atoi(get_substring(line, 1, 2).c_str());
      }

      // Number of spectators
      else if (line.find("<Spectators NSpec=") != std::string::npos) {
          NSpec = std::atoi(get_substring(line, 1, 2).c_str());
      }

      // If variable
      else if (line.find("<Variable VarIndex=") != std::string::npos) {
          varNames.push_back(get_substring(line, 3, 4));
          dumbVars.push_back(0);
      }

      // If spectator
      else if (line.find("Spectator SpecIndex=") != std::string::npos) {
          specNames.push_back(get_substring(line, 3, 4));
          dumbSpecs.push_back(0);
      }
  }

  //
  // Create the reader
  //
  TMVA::Reader tmpTMVAReader( "!Color:Silent:!Error" );

  //
  // Configure all variables and spectators. Note: the order and names
  // must match what is found in the xml weights file!
  //
  for(size_t i = 0; i < NVar; ++i){
      tmpTMVAReader.AddVariable(varNames[i], &dumbVars[i]);
  }

  for(size_t i = 0; i < NSpec; ++i){
      tmpTMVAReader.AddSpectator(specNames[i], &dumbSpecs[i]);
  }

  //
  // Book the method and set up the weights file
  //
  tmpTMVAReader.BookMVA(method , weightFile.fullPath());

  return std::make_unique<const GBRForest>(GBRForest( dynamic_cast<TMVA::MethodBDT*>( tmpTMVAReader.FindMVA(method) ) ));
}
