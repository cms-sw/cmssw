#ifndef HiggsAnalysis_CombinedLimit_Combine_h
#define HiggsAnalysis_CombinedLimit_Combine_h
#include <TString.h>
class TDirectory;
class TTree;
class LimitAlgo;

extern Float_t t_cpu_, t_real_;
//RooWorkspace *writeToysHere = 0;
extern TDirectory *writeToysHere;
extern TDirectory *readToysFromHere;
extern LimitAlgo * algo ;
extern bool verbose;
extern bool withSystematics;

void doCombination(TString hlfFile, const std::string &dataset, double &limit, int &iToy, TTree *tree, int nToys=0);

#endif
