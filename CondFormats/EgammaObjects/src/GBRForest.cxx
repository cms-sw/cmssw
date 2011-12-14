#include "../interface/GBRForest.h"
//#include <iostream>
#include "TMVA/DecisionTree.h"
#include "TMVA/MethodBDT.h"



//_______________________________________________________________________
GBRForest::GBRForest() : 
  fInitialResponse(0.)
{

}

//_______________________________________________________________________
GBRForest::~GBRForest() 
{
}

//_______________________________________________________________________
GBRForest::GBRForest(const TMVA::MethodBDT *bdt) : 
  fInitialResponse(bdt->GetBoostWeights().front())
{
  
  const std::vector<TMVA::DecisionTree*> &forest = bdt->GetForest();
  fTrees.resize(forest.size());
  unsigned int there=0;
  for (std::vector<TMVA::DecisionTree*>::const_iterator it=forest.begin(); it!=forest.end(); ++it) {
    fTrees[there++]=GBRTree(*it);
  }
}

GBRForest::GBRForest(const GBRForest &other) :
  fInitialResponse(other.fInitialResponse)
{
  fTrees.resize(other.fTrees.size());
  unsigned int there=0;
  for (std::vector<GBRTree>::const_iterator it = other.fTrees.begin(); it!=other.fTrees.end(); ++it) {
    fTrees[there++]=GBRTree(*it);
  }
} 



