#include "CondFormats/EgammaObjects/interface/GBRTree.h"

//_______________________________________________________________________
GBRTree::GBRTree()
{

}

bool GBRTree::isTerminal(tinyxml2::XMLElement* node)
{
  bool is = true;
  for(tinyxml2::XMLElement* e = node->FirstChildElement("Node");
          e != nullptr; e = e->NextSiblingElement("Node")) {
      is = false;
  }
  return is;
}

GBRTree::GBRTree(tinyxml2::XMLElement* binaryTree, double scale, bool isregression, bool useyesnoleaf, bool adjustboundary, bool isadaclassifier)
{
  tinyxml2::XMLElement* root = binaryTree->FirstChildElement("Node");

  int nIntermediate = CountIntermediateNodes(root);
  int nTerminal = CountTerminalNodes(root);
  
  //special case, root node is terminal
  if (nIntermediate==0) nIntermediate = 1;
  
  fCutIndices.reserve(nIntermediate);
  fCutVals.reserve(nIntermediate);
  fLeftIndices.reserve(nIntermediate);
  fRightIndices.reserve(nIntermediate);
  fResponses.reserve(nTerminal);

  AddNode(root, scale, isregression, useyesnoleaf, adjustboundary, isadaclassifier);

  //special case, root node is terminal, create fake intermediate node at root
  if (fCutIndices.empty()) {
    fCutIndices.push_back(0);
    fCutVals.push_back(0);
    fLeftIndices.push_back(0);
    fRightIndices.push_back(0);
  }

}

//_______________________________________________________________________
GBRTree::~GBRTree() {

}

//_______________________________________________________________________
unsigned int GBRTree::CountIntermediateNodes(tinyxml2::XMLElement* node) {

  unsigned int count = 0;
  for(tinyxml2::XMLElement* e = node->FirstChildElement("Node");
          e != nullptr; e = e->NextSiblingElement("Node")) {
      count += CountIntermediateNodes(e);
  }
  return count > 0 ? count + 1 : 0;
  
}

//_______________________________________________________________________
unsigned int GBRTree::CountTerminalNodes(tinyxml2::XMLElement* node) {
  
  unsigned int count = 0;
  for(tinyxml2::XMLElement* e = node->FirstChildElement("Node");
          e != nullptr; e = e->NextSiblingElement("Node")) {
      count += CountTerminalNodes(e);
  }
  return count > 0 ? count : 1;
  
}

//_______________________________________________________________________
void GBRTree::AddNode(tinyxml2::XMLElement* node, double scale, bool isregression, bool useyesnoleaf, bool adjustboundary, bool isadaclassifier) {

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
    fResponses.push_back(response);
    return;
  }
  else {    
    int thisidx = fCutIndices.size();
    
    int selector; 
    float cutval; 
    bool ctype; 

    node->QueryIntAttribute("IVar", &selector);
    node->QueryFloatAttribute("Cut", &cutval);
    node->QueryBoolAttribute("cType", &ctype);

    fCutIndices.push_back(static_cast<unsigned char>(selector));
    //newer tmva versions use >= instead of > in decision tree splits, so adjust cut value
    //to reproduce the correct behaviour
    if (adjustboundary) {
      cutval = std::nextafter(cutval,std::numeric_limits<float>::lowest());
    }
    fCutVals.push_back(cutval);
    fLeftIndices.push_back(0);   
    fRightIndices.push_back(0);
    
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
    
    bool leftIsTerminal = isTerminal(left);
    bool rightIsTerminal = isTerminal(right);

    if (leftIsTerminal) {
      fLeftIndices[thisidx] = -fResponses.size();
    }
    else {
      fLeftIndices[thisidx] = fCutIndices.size();
    }
    AddNode(left, scale, isregression, useyesnoleaf, adjustboundary,isadaclassifier);
    
    if (rightIsTerminal) {
      fRightIndices[thisidx] = -fResponses.size();
    }
    else {
      fRightIndices[thisidx] = fCutIndices.size();
    }
    AddNode(right, scale, isregression, useyesnoleaf, adjustboundary,isadaclassifier);
    
  }
  
}
