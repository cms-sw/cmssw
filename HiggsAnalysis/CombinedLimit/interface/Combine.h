#ifndef HiggsAnalysis_CombinedLimit_Combine_h
#define HiggsAnalysis_CombinedLimit_Combine_h
#include <TString.h>
class TDirectory;
class TTree;

extern Float_t t_cpu_, t_real_;
//RooWorkspace *writeToysHere = 0;
extern TDirectory *writeToysHere;
extern TDirectory *readToysFromHere;

enum MethodType {
  undefined, hybrid, profileLikelihood, bayesianFlatPrior, mcmc, mcmcUniform
};

extern MethodType method;

void doCombination(TString hlfFile, double &limit, int &iToy, TTree *tree, int nToys=0, bool withSystematics=true);

#endif
