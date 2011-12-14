#include "../interface/GBRForest.h"
//#include <iostream>
#include "TMVA/DecisionTree.h"
#include "TMVA/MethodBDT.h"



//_______________________________________________________________________
GBRForest::GBRForest() : 
  fInitialResponse(0.),
  fTrees(0)
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
  fTrees.reserve(forest.size());
  std::vector<TMVA::DecisionTree*>::const_iterator fBegin = forest.begin();
  std::vector<TMVA::DecisionTree*>::const_iterator fEnd = forest.end();
  for (std::vector<TMVA::DecisionTree*>::const_iterator it=fBegin; it!=fEnd; ++it) {
    fTrees.push_back(GBRTree(*it));
  }
  
}

GBRForest::GBRForest(const GBRForest &other) :
  fInitialResponse(other.fInitialResponse)
{
  
  fTrees.reserve(other.fTrees.size());
  std::vector<GBRTree>::const_iterator fBegin = other.fTrees.begin();
  std::vector<GBRTree>::const_iterator fEnd = other.fTrees.end();
  for (std::vector<GBRTree>::const_iterator it=fBegin; it!=fEnd; ++it) {
    fTrees.push_back(GBRTree(*it));
  }
  
} 



