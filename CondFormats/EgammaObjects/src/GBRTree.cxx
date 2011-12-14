

#include "CondFormats/EgammaObjects/interface/GBRTree.h"
#include "TClass.h"

using namespace std;
#include "TMVA/DecisionTreeNode.h"
#include "TMVA/DecisionTree.h"


//ClassImp(GBRTree)


//_______________________________________________________________________
GBRTree::GBRTree() : 
  fNIntermediateNodes(0),
  fNTerminalNodes(0)
{

}

//_______________________________________________________________________
GBRTree::GBRTree(const TMVA::DecisionTree *tree) : 
  fNIntermediateNodes(0),
  fNTerminalNodes(0)
{
  
  //printf("boostweights size = %i, forest size = %i\n",bdt->GetBoostWeights().size(),bdt->GetForest().size());
  Int_t nIntermediate = CountIntermediateNodes((TMVA::DecisionTreeNode*)tree->GetRoot());
  Int_t nTerminal = CountTerminalNodes((TMVA::DecisionTreeNode*)tree->GetRoot());
  
  //special case, root node is terminal
  if (nIntermediate==0) nIntermediate = 1;
  
  fCutIndices.resize(nIntermediate);
  fCutVals.resize(nIntermediate);
  fLeftIndices.resize(nIntermediate);
  fRightIndices.resize(nIntermediate);
  fResponses.resize(nTerminal);

  AddNode((TMVA::DecisionTreeNode*)tree->GetRoot());

  //special case, root node is terminal, create fake intermediate node at root
  if (fNIntermediateNodes==0) {
    fNIntermediateNodes=1;
    fCutIndices.resize(nIntermediate);
    fCutVals.resize(nIntermediate);
    fLeftIndices.resize(nIntermediate);
    fRightIndices.resize(nIntermediate);
    fResponses.resize(nTerminal);
    
    fCutIndices[0] = 0;
    fCutVals[0] = 0.;
    fLeftIndices[0] = 0;
    fRightIndices[0] = 0;
  }

}

//_______________________________________________________________________
GBRTree::~GBRTree() {

}

//_______________________________________________________________________
UInt_t GBRTree::CountIntermediateNodes(const TMVA::DecisionTreeNode *node) {
  
  if (!node->GetLeft() || !node->GetRight() || node->IsTerminal()) {
    return 0;
  }
  else {
    return 1 + CountIntermediateNodes((TMVA::DecisionTreeNode*)node->GetLeft()) + CountIntermediateNodes((TMVA::DecisionTreeNode*)node->GetRight());
  }
  
}

//_______________________________________________________________________
UInt_t GBRTree::CountTerminalNodes(const TMVA::DecisionTreeNode *node) {
  
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
    fResponses[fNTerminalNodes] = node->GetResponse();
    ++fNTerminalNodes;
    return;
  }
  else {    
    Int_t thisindex = fNIntermediateNodes;
    ++fNIntermediateNodes;
    
    fCutIndices[thisindex] = node->GetSelector();
    fCutVals[thisindex] = node->GetCutValue();

    
    
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
      fLeftIndices[thisindex] = -fNTerminalNodes;
    }
    else {
      fLeftIndices[thisindex] = fNIntermediateNodes;
    }
    AddNode(left);
    
    if (!right->GetLeft() || !right->GetRight() || right->IsTerminal()) {
      fRightIndices[thisindex] = -fNTerminalNodes;
    }
    else {
      fRightIndices[thisindex] = fNIntermediateNodes;
    }
    AddNode(right);    
    
  }
  
}
