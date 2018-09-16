#include "CommonTools/MVAUtils/interface/GBRForestTools.h"
#include "CommonTools/MVAUtils/interface/TMVAZipReader.h"
#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"

#include <cstdio>
#include <cstdlib>
#include <RVersion.h>
#include <cmath>
#include <tinyxml2.h>


size_t readVariables(tinyxml2::XMLElement* root, const char * key, std::vector<std::string>& names)
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

bool isTerminal(tinyxml2::XMLElement* node)
{
  bool is = true;
  for(tinyxml2::XMLElement* e = node->FirstChildElement("Node");
          e != nullptr; e = e->NextSiblingElement("Node")) {
      is = false;
  }
  return is;
}

unsigned int countIntermediateNodes(tinyxml2::XMLElement* node) {

  unsigned int count = 0;
  for(tinyxml2::XMLElement* e = node->FirstChildElement("Node");
          e != nullptr; e = e->NextSiblingElement("Node")) {
      count += countIntermediateNodes(e);
  }
  return count > 0 ? count + 1 : 0;
  
}

unsigned int countTerminalNodes(tinyxml2::XMLElement* node) {
  
  unsigned int count = 0;
  for(tinyxml2::XMLElement* e = node->FirstChildElement("Node");
          e != nullptr; e = e->NextSiblingElement("Node")) {
      count += countTerminalNodes(e);
  }
  return count > 0 ? count : 1;
  
}

void addNode(GBRTree& tree, tinyxml2::XMLElement* node, double scale, bool isregression, bool useyesnoleaf, bool adjustboundary, bool isadaclassifier) {

  bool nodeIsTerminal = isTerminal(node);
  if (nodeIsTerminal) {
    double response = 0.;
    if (isregression) {
      node->QueryDoubleAttribute("res", &response);
    }
    else {
      if (useyesnoleaf) {
        node->QueryDoubleAttribute("nType", &response);
      }
      else {
        if (isadaclassifier) {
            node->QueryDoubleAttribute("purity", &response);
        } else {
            node->QueryDoubleAttribute("res", &response);
        }
      }
    }
    response *= scale;
    tree.Responses().push_back(response);
  }
  else {    

    int thisidx = tree.CutIndices().size(); 
    
    int selector; 
    float cutval; 
    bool ctype; 

    node->QueryIntAttribute("IVar", &selector);
    node->QueryFloatAttribute("Cut", &cutval);
    node->QueryBoolAttribute("cType", &ctype);

    tree.CutIndices().push_back(static_cast<unsigned char>(selector)); 

    //newer tmva versions use >= instead of > in decision tree splits, so adjust cut value
    //to reproduce the correct behaviour
    if (adjustboundary) {
      cutval = std::nextafter(cutval,std::numeric_limits<float>::lowest());
    }
    tree.CutVals().push_back(cutval);
    tree.LeftIndices().push_back(0);   
    tree.RightIndices().push_back(0);
    
    tinyxml2::XMLElement* left = nullptr;
    tinyxml2::XMLElement* right = nullptr;
    for(tinyxml2::XMLElement* e = node->FirstChildElement("Node");
            e != nullptr; e = e->NextSiblingElement("Node")) {
        if (*(e->Attribute("pos")) == 'l') left = e;
        else if (*(e->Attribute("pos")) == 'r') right = e;
    }
    if (!ctype) {
      std::swap(left, right);
    }

    tree.LeftIndices()[thisidx] = isTerminal(left) ? -tree.Responses().size() : tree.CutIndices().size() ;
    addNode(tree, left, scale, isregression, useyesnoleaf, adjustboundary,isadaclassifier);
    
    tree.RightIndices()[thisidx] = isTerminal(right) ? -tree.Responses().size() : tree.CutIndices().size() ;
    addNode(tree, right, scale, isregression, useyesnoleaf, adjustboundary,isadaclassifier);
    
  }
  
}

std::unique_ptr<GBRForest> init(const std::string& weightsFileFullPath, std::vector<std::string>& varNames) {

  std::string method;
  //
  // Load weights file, for gzipped or raw xml file
  //
  tinyxml2::XMLDocument xmlDoc;

  if (reco::details::hasEnding(weightsFileFullPath, ".xml")) {
      xmlDoc.LoadFile(weightsFileFullPath.c_str());
  } else if (reco::details::hasEnding(weightsFileFullPath, ".gz") ||
             reco::details::hasEnding(weightsFileFullPath, ".gzip")) {
      xmlDoc.Parse(reco::details::readGzipFile(weightsFileFullPath));
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
  if (info.find("ROOT Release") != info.end()) {
    std::string s = info["ROOT Release"];
    rootTrainingVersion = std::stoi(s.substr(s.find("[")+1,s.find("]")-s.find("[")-1));
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
    
  auto forest = std::make_unique<GBRForest>();
  forest->SetInitialResponse(isregression ? boostWeights[0] : 0.);
  
  double norm = 0;
  if (isadaclassifier) {
    for (double w : boostWeights) {
      norm += w;
    }
  }

  forest->Trees().reserve(boostWeights.size());
  size_t itree = 0;
  // Loop over tree estimators
  for(tinyxml2::XMLElement* e = weightsElem->FirstChildElement("BinaryTree");
          e != nullptr; e = e->NextSiblingElement("BinaryTree")) {
    double scale = isadaclassifier ? boostWeights[itree]/norm : 1.0;

    tinyxml2::XMLElement* root = e->FirstChildElement("Node");
    forest->Trees().push_back(GBRTree(countIntermediateNodes(root), countTerminalNodes(root)));
    auto & tree = forest->Trees().back();

    addNode(tree, root, scale, isregression, useyesnoleaf, adjustboundaries, isadaclassifier);

    //special case, root node is terminal, create fake intermediate node at root
    if (tree.CutIndices().empty()) {
      tree.CutIndices().push_back(0);
      tree.CutVals().push_back(0);
      tree.LeftIndices().push_back(0);
      tree.RightIndices().push_back(0);
    }

    ++itree;
  }

  return forest;
}

// Create a GBRForest from an XML weight file
std::unique_ptr<const GBRForest>
createGBRForest(const std::string     &weightFile)
{
    std::vector<std::string> varNames;
    return createGBRForest(weightFile, varNames);
}

std::unique_ptr<const GBRForest>
createGBRForest(const edm::FileInPath &weightFile)
{
    std::vector<std::string> varNames;
    return createGBRForest(weightFile.fullPath(), varNames);
}

// Overloaded versions which are taking string vectors by reference to strore the variable names in
std::unique_ptr<const GBRForest>
createGBRForest(const std::string     &weightFile, std::vector<std::string> &varNames)
{
    std::unique_ptr<GBRForest> gbrForest;

    if(weightFile[0] == '/') {
        gbrForest = init(weightFile, varNames);
    }
    else {
        edm::FileInPath weightFileEdm(weightFile);
        gbrForest = init( weightFileEdm.fullPath(), varNames);
    }
    return gbrForest;
}

std::unique_ptr<const GBRForest>
createGBRForest(const edm::FileInPath &weightFile, std::vector<std::string> &varNames)
{
    return createGBRForest(weightFile.fullPath(), varNames);
}
