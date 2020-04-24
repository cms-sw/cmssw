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
GBRForest::GBRForest(const TMVA::MethodBDT *bdt)
{
  
  //special handling for non-gradient-boosted (ie ADABoost) classifiers, where tree responses
  //need to be renormalized after the training for evaluation purposes
  bool isadaclassifier = !bdt->DoRegression() && !bdt->GetOptions().Contains("~BoostType=Grad");  
  bool useyesnoleaf = isadaclassifier && bdt->GetOptions().Contains("~UseYesNoLeaf=True");
  bool isregression = bdt->DoRegression();
  //newer tmva versions use >= instead of > in decision tree splits, so adjust cut value
  //to reproduce the correct behaviour  
  bool adjustboundaries = (bdt->GetTrainingROOTVersionCode()>=ROOT_VERSION(5,34,20) && bdt->GetTrainingROOTVersionCode()<ROOT_VERSION(6,0,0)) || bdt->GetTrainingROOTVersionCode()>=ROOT_VERSION(6,2,0);
    
  if (isregression) {
    fInitialResponse = bdt->GetBoostWeights().front();
  }
  else {
    fInitialResponse = 0.;
  }
  
  double norm = 0;
  if (isadaclassifier) {
    for (std::vector<double>::const_iterator it=bdt->GetBoostWeights().begin(); it!=bdt->GetBoostWeights().end(); ++it) {
      norm += *it;  
    }
  }
  
  const std::vector<TMVA::DecisionTree*> &forest = bdt->GetForest();
  fTrees.reserve(forest.size());
  for (unsigned int itree=0; itree<forest.size(); ++itree) {
    double scale = isadaclassifier ? bdt->GetBoostWeights()[itree]/norm : 1.0;
    fTrees.push_back(GBRTree(forest[itree],scale,useyesnoleaf,adjustboundaries));
  }
  
}




