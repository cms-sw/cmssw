#ifndef __L1Analysis_L1AnalysisRecoVertex_H__
#define __L1Analysis_L1AnalysisRecoVertex_H__

#include <TChain.h>
#include <iostream>
#include <vector>

namespace L1Analysis
{
  class L1AnalysisRecoVertex
  {
    
  public :
    void initTree(TChain * tree, const std::string & className);
  
  public:
    L1AnalysisRecoVertex() {}
    void print();
    
    // ---- General L1AnalysisRecoVertex information.    
    unsigned int nVtx;
    std::vector<unsigned int> vtxNDoF;
    std::vector<double>       vtxZ;
    std::vector<double>       vtxRho;
};
}


#endif

#ifdef l1ntuple_cxx


void L1Analysis::L1AnalysisRecoVertex::initTree(TChain * tree, const std::string & className)
{
  SetBranchAddress(tree,"nVtx",    className, &nVtx );
  SetBranchAddress(tree,"vtxNDoF", className, &vtxNDoF);
  SetBranchAddress(tree,"vtxZ",    className, &vtxZ);
  SetBranchAddress(tree,"vtxRho",  className, &vtxRho);
}


void L1Analysis::L1AnalysisRecoVertex::print()
{
}

#endif


