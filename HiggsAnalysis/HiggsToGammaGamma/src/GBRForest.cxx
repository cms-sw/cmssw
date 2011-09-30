


#include "../interface/GBRForest.h"
//#include <iostream>
#include "TMVA/DecisionTree.h"
#include "TMVA/MethodBDT.h"


ClassImp(GBRForest)


//_______________________________________________________________________
GBRForest::GBRForest() : 
  fInitialResponse(0.)
{

}

//_______________________________________________________________________
GBRForest::~GBRForest() 
{
  for (UInt_t i=0; i<fTrees.size(); ++i) {
    delete fTrees.at(i);
  }
}

//_______________________________________________________________________
GBRForest::GBRForest(const TMVA::MethodBDT *bdt) : 
  TNamed("GBRForest","GBRForest"),
  fInitialResponse(bdt->GetBoostWeights().front())
{
  
  const std::vector<TMVA::DecisionTree*> &forest = bdt->GetForest();
  for (std::vector<TMVA::DecisionTree*>::const_iterator it=forest.begin(); it!=forest.end(); ++it) {
    GBRTree *tree = new GBRTree(*it);
    fTrees.push_back(tree);
  }
}



