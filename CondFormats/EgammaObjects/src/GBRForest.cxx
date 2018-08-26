#include "CondFormats/EgammaObjects/interface/GBRForest.h"

#include "FWCore/Utilities/interface/Exception.h"
#include <cstdio>
#include <cstdlib>
#include <cstdio>
#include <cstdlib>
#include "zlib.h"

#include <iostream>

#define ROOT_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))

namespace {

    // From http://stackoverflow.com/questions/874134/find-if-string-endswith-another-string-in-c
    bool hasEnding(std::string const &fullString, std::string const &ending) {
      if (fullString.length() >= ending.length()) {
        return (0 == fullString.compare(
              fullString.length() - ending.length(), ending.length(), ending));
      } else {
        return false;
      }
    }

    char* readGzipFile(const std::string& weightsFile)
    {
      FILE *f = fopen(weightsFile.c_str(), "r");
      if (f==nullptr) {
        throw cms::Exception("InvalidFileState")
          << "Failed to open MVA file = " << weightsFile << " !!\n";
      }
      int magic;
      int size;
      fread(&magic, 4, 1, f);
      fseek(f, -4, SEEK_END);
      fread(&size, 4, 1, f);
      fclose(f);
      //printf("%x, %i\n", magic, size);

      gzFile  file = gzopen (weightsFile.c_str(), "r");

      int bytes_read;
      char *buffer = (char*)malloc(size);
      bytes_read = gzread (file, buffer, size - 1);
      buffer[bytes_read] = '\0';
      if (!gzeof (file)) {
        int err;
        const char * error_string;
        error_string = gzerror (file, & err);
        if (err) {
          free(buffer);
          throw cms::Exception("InvalidFileState")
            << "Error while reading gzipped file = " << weightsFile << " !!\n"
                << error_string;
        }
      }
      gzclose (file);
      return buffer;
    }
};

//_______________________________________________________________________
GBRForest::GBRForest() : 
  fInitialResponse(0.)
{}

GBRForest::GBRForest(const std::string& weightsFile)
{
  std::vector<std::string> varNames;
  if(weightsFile[0] == '/') {
      init(weightsFile, varNames);
  }
  else {
      edm::FileInPath weightsFileEdm(weightsFile);
      init(weightsFileEdm.fullPath(), varNames);
  }
}

GBRForest::GBRForest(const edm::FileInPath& weightsFile)
{
  std::vector<std::string> varNames;
  init(weightsFile.fullPath(), varNames);
}

GBRForest::GBRForest(const std::string& weightsFile, std::vector<std::string>& varNames)
{
  if(weightsFile[0] == '/') {
      init(weightsFile, varNames);
  }
  else {
      edm::FileInPath weightsFileEdm(weightsFile);
      init(weightsFileEdm.fullPath(), varNames);
  }
}

GBRForest::GBRForest(const edm::FileInPath& weightsFile, std::vector<std::string>& varNames)
{
  init(weightsFile.fullPath(), varNames);
}

void GBRForest::init(const std::string& weightsFileFullPath, std::vector<std::string>& varNames) {

  std::string method;
  //
  // Load weights file, for gzipped or raw xml file
  //
  tinyxml2::XMLDocument xmlDoc;

  if (hasEnding(weightsFileFullPath, ".xml")) {
      xmlDoc.LoadFile(weightsFileFullPath.c_str());
  } else if (hasEnding(weightsFileFullPath, ".gz") ||
             hasEnding(weightsFileFullPath, ".gzip")) {
      xmlDoc.Parse(readGzipFile(weightsFileFullPath));
  }

  //tinyxml2::XMLHandle xmlDoc( &xmlDoc );

  tinyxml2::XMLElement* root = xmlDoc.FirstChildElement("MethodSetup");
  method = root->Attribute("Method");
  readVariables(root->FirstChildElement("Variables"), "Variable", varNames);

  // Read in the TMVA general info
  std::map <std::string, std::string> info; 
  tinyxml2::XMLElement* infoElem = xmlDoc.FirstChildElement("MethodSetup")->FirstChildElement("GeneralInfo");
  if (infoElem == nullptr) {
      throw cms::Exception("XMLError")
          << "No GeneralInfo found in " << weightsFileFullPath << " !!\n";
  }
  for(tinyxml2::XMLElement* e = infoElem->FirstChildElement("Info");
          e != nullptr; e = e->NextSiblingElement("Info"))
  {
      const char * name;
      const char * value;
      e->QueryStringAttribute("name",  &name);
      e->QueryStringAttribute("value", &value);
      info[name] = value;
  }

  // Read in the TMVA options
  std::map <std::string, std::string> options; 
  tinyxml2::XMLElement* optionsElem = xmlDoc.FirstChildElement("MethodSetup")->FirstChildElement("Options");
  if (optionsElem == nullptr) {
      throw cms::Exception("XMLError")
          << "No Options found in " << weightsFileFullPath << " !!\n";
  }
  for(tinyxml2::XMLElement* e = optionsElem->FirstChildElement("Option");
          e != nullptr; e = e->NextSiblingElement("Option"))
  {
      const char * name;
      e->QueryStringAttribute("name",  &name);
      options[name] = e->GetText();
  }

  // Get root version number if available
  int rootTrainingVersion(0);
  if (options.find("ROOT Release") != options.end()) {
    std::string s = options["ROOT Release"];
    rootTrainingVersion = std::stoi(s.substr(s.find("[")+1,s.find("]")-s.find("[")));
  }

  // Get the boosting weights
  std::vector<double> boostWeights;
  tinyxml2::XMLElement* weightsElem = xmlDoc.FirstChildElement("MethodSetup")->FirstChildElement("Weights");
  if (weightsElem == nullptr) {
      throw cms::Exception("XMLError")
          << "No Weights found in " << weightsFileFullPath << " !!\n";
  }
  bool hasTrees = false;
  for(tinyxml2::XMLElement* e = weightsElem->FirstChildElement("BinaryTree");
          e != nullptr; e = e->NextSiblingElement("BinaryTree"))
  {
      hasTrees = true;
      double w;
      e->QueryDoubleAttribute("boostWeight", &w);
      boostWeights.push_back(w);
  }
  if (!hasTrees) {
      throw cms::Exception("XMLError")
          << "No BinaryTrees found in " << weightsFileFullPath << " !!\n";
  }

  bool isregression = info["AnalysisType"] == "Regression";

  //special handling for non-gradient-boosted (ie ADABoost) classifiers, where tree responses
  //need to be renormalized after the training for evaluation purposes
  bool isadaclassifier = !isregression && options["BoostType"] != "Grad";
  bool useyesnoleaf = isadaclassifier && options["UseYesNoLeaf"] == "True";

  //newer tmva versions use >= instead of > in decision tree splits, so adjust cut value
  //to reproduce the correct behaviour  
  bool adjustboundaries = (rootTrainingVersion>=ROOT_VERSION(5,34,20) && rootTrainingVersion<ROOT_VERSION(6,0,0)) || rootTrainingVersion>=ROOT_VERSION(6,2,0);
    
  if (isregression) {
    fInitialResponse = boostWeights[0];
  }
  else {
    fInitialResponse = 0.;
  }
  
  double norm = 0;
  if (isadaclassifier) {
    for (double w : boostWeights) {
      norm += w;
    }
  }

  fTrees.reserve(boostWeights.size());
  size_t itree = 0;
  // Loop over tree estimators
  for(tinyxml2::XMLElement* e = weightsElem->FirstChildElement("BinaryTree");
          e != nullptr; e = e->NextSiblingElement("BinaryTree")) {
    double scale = isadaclassifier ? boostWeights[itree]/norm : 1.0;
    fTrees.push_back(GBRTree(e,scale,isregression,useyesnoleaf,adjustboundaries));
    ++itree;
  }
}

//_______________________________________________________________________
GBRForest::~GBRForest() 
{}

size_t GBRForest::readVariables(tinyxml2::XMLElement* root, const char * key, std::vector<std::string>& names)
{
  size_t n = 0;
  names.clear();

  if (root != nullptr) {
      for(tinyxml2::XMLElement* e = root->FirstChildElement(key);
              e != nullptr; e = e->NextSiblingElement(key))
      {
          names.push_back(e->Attribute("Expression"));
          ++n;
      }
  }

  return n;
}
