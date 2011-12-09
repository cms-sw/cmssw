

#include "CondFormats/EgammaObjects/interface/GBRTree.h"
#include "TClass.h"

using namespace std;
#include "TMVA/DecisionTreeNode.h"
#include "TMVA/DecisionTree.h"


//ClassImp(GBRTree)


//_______________________________________________________________________
GBRTree::GBRTree() : 
  fNIntermediateNodes(0),
  fNTerminalNodes(0),
  fCutIndices(0),
  fCutVals(0),
  fLeftIndices(0),
  fRightIndices(0),
  fResponses(0)
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

  /*  
  fCutIndices = new UChar_t[nIntermediate];
  fCutVals = new Float_t[nIntermediate];
  fLeftIndices = new Int_t[nIntermediate];
  fRightIndices = new Int_t[nIntermediate];
  fResponses = new Float_t[nTerminal];
  */
  
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

/*
//_______________________________________________________________________
GBRTree::GBRTree(const GBRTree &other) :
  fNIntermediateNodes(other.fNIntermediateNodes),
  fNTerminalNodes(other.fNTerminalNodes)
{
  fCutIndices = new UChar_t[fNIntermediateNodes];
  fCutVals = new Float_t[fNIntermediateNodes];
  fLeftIndices = new Int_t[fNIntermediateNodes];
  fRightIndices = new Int_t[fNIntermediateNodes];
  fResponses = new Float_t[fNTerminalNodes];

  for (Int_t i=0; i<fNIntermediateNodes; ++i) {
    fCutIndices[i]   =  other.fCutIndices[i];
    fCutVals[i]      =  other.fCutVals[i];
    fLeftIndices[i]  =  other.fLeftIndices[i];
    fRightIndices[i] =  other.fRightIndices[i];
  }

  for (Int_t i=0; i<fNTerminalNodes; ++i) {
    fResponses[i]    =  other.fResponses[i];
  }

}
*/

//_______________________________________________________________________
GBRTree::~GBRTree() {
  /*
    delete [] fCutIndices;
    delete [] fCutVals;
    delete [] fLeftIndices;
    delete [] fRightIndices;
    delete [] fResponses;
  */
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


//-------------------------------------------------------------------------------------------------
/*
void GBRTree::Streamer(TBuffer &b)
{
   // Stream all objects in the array to or from the I/O buffer.
   // Ugly special case handling for Double32

  if (b.IsReading()) {

    Version_t v = b.ReadVersion(0,0,GBRTree::Class());

    if (v<=1) {
      UInt_t start=0;
      UInt_t count=0;
      b.ReadClassBuffer(GBRTree::Class(),this,v,start,count);
      return;    
    }

    b >> fNIntermediateNodes;
    b >> fNTerminalNodes;

    if (fNIntermediateNodes) {
      fCutIndices = new UChar_t[fNIntermediateNodes];
      fCutVals = new Float_t[fNIntermediateNodes];
      fLeftIndices = new Int_t[fNIntermediateNodes];
      fRightIndices = new Int_t[fNIntermediateNodes];
     
      b.ReadFastArray(fCutIndices,fNIntermediateNodes);
      b.ReadFastArray(fCutVals,fNIntermediateNodes);
      b.ReadFastArray(fLeftIndices,fNIntermediateNodes);
      b.ReadFastArray(fRightIndices,fNIntermediateNodes);
    }

    if (fNTerminalNodes) {
      fResponses = new Float_t[fNTerminalNodes];
      b.ReadFastArray(fResponses,fNTerminalNodes);
    }

  } else { //writing

    b.WriteVersion(GBRTree::Class());
    b << fNIntermediateNodes;
    b << fNTerminalNodes;

    if (fNIntermediateNodes) {
      b.WriteFastArray(fCutIndices,fNIntermediateNodes);
      b.WriteFastArray(fCutVals,fNIntermediateNodes);
      b.WriteFastArray(fLeftIndices,fNIntermediateNodes);
      b.WriteFastArray(fRightIndices,fNIntermediateNodes);
    }
    if (fNTerminalNodes) {
      b.WriteFastArray(fResponses,fNTerminalNodes);
    }
  }
}
*/
