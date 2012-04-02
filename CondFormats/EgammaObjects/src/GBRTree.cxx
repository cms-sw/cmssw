

#include "CondFormats/EgammaObjects/interface/GBRTree.h"

using namespace std;
#include "TMVA/DecisionTreeNode.h"
#include "TMVA/DecisionTree.h"

//_______________________________________________________________________
GBRTree::GBRTree()
{

}

//_______________________________________________________________________
GBRTree::GBRTree(const TMVA::DecisionTree *tree)
{
  
  //printf("boostweights size = %i, forest size = %i\n",bdt->GetBoostWeights().size(),bdt->GetForest().size());
  int nIntermediate = CountIntermediateNodes((TMVA::DecisionTreeNode*)tree->GetRoot());
  int nTerminal = CountTerminalNodes((TMVA::DecisionTreeNode*)tree->GetRoot());
  
  //special case, root node is terminal
  if (nIntermediate==0) nIntermediate = 1;
  
  fCutIndices.reserve(nIntermediate);
  fCutVals.reserve(nIntermediate);
  fLeftIndices.reserve(nIntermediate);
  fRightIndices.reserve(nIntermediate);
  fResponses.reserve(nTerminal);

  AddNode((TMVA::DecisionTreeNode*)tree->GetRoot());

  //special case, root node is terminal, create fake intermediate node at root
  if (fCutIndices.size()==0) {
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
unsigned int GBRTree::CountIntermediateNodes(const TMVA::DecisionTreeNode *node) {
  
  if (!node->GetLeft() || !node->GetRight() || node->IsTerminal()) {
    return 0;
  }
  else {
    return 1 + CountIntermediateNodes((TMVA::DecisionTreeNode*)node->GetLeft()) + CountIntermediateNodes((TMVA::DecisionTreeNode*)node->GetRight());
  }
  
}

//_______________________________________________________________________
unsigned int GBRTree::CountTerminalNodes(const TMVA::DecisionTreeNode *node) {
  
  if (!node->GetLeft() || !node->GetRight() || node->IsTerminal()) {
    return 1;
  }
  else {
    return 0 + CountTerminalNodes((TMVA::DecisionTreeNode*)node->GetLeft()) + CountTerminalNodes((TMVA::DecisionTreeNode*)node->GetRight());
  }
  
}


//_______________________________________________________________________
void GBRTree::AddNode(const TMVA::DecisionTreeNode *node) {

  if (!node->GetLeft() || !node->GetRight() || node->IsTerminal()) {
    fResponses.push_back(node->GetResponse());
    return;
  }
  else {    
    int thisidx = fCutIndices.size();
    
    fCutIndices.push_back(node->GetSelector());
    fCutVals.push_back(node->GetCutValue());
    fLeftIndices.push_back(0);   
    fRightIndices.push_back(0);
    
    TMVA::DecisionTreeNode *left;
    TMVA::DecisionTreeNode *right;
    if (node->GetCutType()) {
      left = (TMVA::DecisionTreeNode*)node->GetLeft();
      right = (TMVA::DecisionTreeNode*)node->GetRight();
    }
    else {
      left = (TMVA::DecisionTreeNode*)node->GetRight();
      right = (TMVA::DecisionTreeNode*)node->GetLeft();
    }
    
    
    if (!left->GetLeft() || !left->GetRight() || left->IsTerminal()) {
      fLeftIndices[thisidx] = -fResponses.size();
    }
    else {
      fLeftIndices[thisidx] = fCutIndices.size();
    }
    AddNode(left);
    
    if (!right->GetLeft() || !right->GetRight() || right->IsTerminal()) {
      fRightIndices[thisidx] = -fResponses.size();
    }
    else {
      fRightIndices[thisidx] = fCutIndices.size();
    }
    AddNode(right);    
    
  }
  
}
