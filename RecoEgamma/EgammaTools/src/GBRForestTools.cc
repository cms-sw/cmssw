#include "RecoEgamma/EgammaTools/interface/GBRForestTools.h"

#include <iostream>
#include <fstream>

namespace {

  // Will return position of n-th occurence of a char in a string.
  int strpos(const std::string &haystack, char needle, unsigned int nth)
  {
    int found = 0;
    for (unsigned int i=0 ; i<nth ; ++i) {
        std::size_t pos = haystack.find(needle, found);
        if (pos == std::string::npos) return -1; 
        else found = pos+1;
    }
    return found;
  }

  // To get the substring between the n1th and n2th quotation mark in a string
  std::string get_quoted_substring(const std::string &str, int n1, int n2)
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
  std::ifstream f;
  std::string tmpstr;

  bool gzipped = false;

  //
  // Set up the input buffers, for gzipped or raw xml file
  //
  if (reco::details::hasEnding(weightFile.fullPath(), ".xml")) {
      f.open(weightFile.fullPath());
      tmpstr = "";
  } else if (reco::details::hasEnding(weightFile.fullPath(), ".gz") || reco::details::hasEnding(weightFile.fullPath(), ".gzip")) {
      gzipped = true;
      tmpstr = std::string(reco::details::readGzipFile(weightFile.fullPath()));
  }
  std::stringstream is(tmpstr);

  bool isend;

  while(true) {

      if (gzipped) isend = !std::getline(is, line);
      else isend = !std::getline(f, line);

      if (isend) break;

      // Terminate reading of weights file
      if (line.find("<Weights ") != std::string::npos) break;

      // Method name
      else if (line.find("<MethodSetup Method=") != std::string::npos) {
          method = get_quoted_substring(line, 1, 2);
      }

      // Number of variables
      else if (line.find("<Variables NVar=") != std::string::npos) {
          NVar = std::atoi(get_quoted_substring(line, 1, 2).c_str());
      }

      // Number of spectators
      else if (line.find("<Spectators NSpec=") != std::string::npos) {
          NSpec = std::atoi(get_quoted_substring(line, 1, 2).c_str());
      }

      // If variable
      else if (line.find("<Variable ") != std::string::npos) {
          unsigned int pos = line.find("Expression=");
          varNames.push_back(get_quoted_substring(line.substr(pos, line.length() - pos), 1, 2));
          dumbVars.push_back(0);
      }

      // If spectator
      else if (line.find("Spectator ") != std::string::npos) {
          unsigned int pos = line.find("Expression=");
          specNames.push_back(get_quoted_substring(line.substr(pos, line.length() - pos), 1, 2));
          dumbSpecs.push_back(0);
      }
  }

  //
  // Create the reader
  //
  TMVA::Reader* mvaReader = new TMVA::Reader("!Color:Silent:!Error");

  //
  // Configure all variables and spectators. Note: the order and names
  // must match what is found in the xml weights file!
  //
  for(size_t i = 0; i < NVar; ++i){
      mvaReader->AddVariable(varNames[i], &dumbVars[i]);
  }

  for(size_t i = 0; i < NSpec; ++i){
      mvaReader->AddSpectator(specNames[i], &dumbSpecs[i]);
  }

  //
  // Book the method and set up the weights file
  //

  reco::details::loadTMVAWeights(mvaReader, method, weightFile.fullPath());

  TMVA::MethodBDT* bdt = dynamic_cast<TMVA::MethodBDT*>( mvaReader->FindMVA(method) );
  std::unique_ptr<const GBRForest> gbrForest = std::make_unique<const GBRForest>(GBRForest(bdt));
  delete mvaReader;

  return gbrForest;
}
