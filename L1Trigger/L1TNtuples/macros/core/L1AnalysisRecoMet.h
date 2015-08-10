#ifndef __L1Analysis_L1AnalysisMet_H__
#define __L1Analysis_L1AnalysisMet_H__

#include <TTree.h>
#include <iostream>

namespace L1Analysis
{
  class L1AnalysisRecoMet
{

  public : 
  void initTree(TTree * ttree, const std::string & className);

  public:
  L1AnalysisRecoMet() {}
    
    // ---- General L1AnalysisRecoMet information.
    double met;
    double metPhi;
    double Ht;
    double mHt;
    double mHtPhi;
    double sumEt;

    void print();
};
}

void L1Analysis::L1AnalysisRecoMet::initTree(TTree * tree, const std::string & className)
{
     SetBranchAddress(tree,"met", className,    &met);
     SetBranchAddress(tree,"metPhi", className, &metPhi);
     SetBranchAddress(tree,"Ht", className,     &Ht);
     SetBranchAddress(tree,"mHt", className,    &mHt);
     SetBranchAddress(tree,"mHtPhi", className, &mHtPhi);
     SetBranchAddress(tree,"sumEt", className,  &sumEt);
}


void L1Analysis::L1AnalysisRecoMet::print()
{
  std::cout << "met="<<met<<" "
            << "metPhi="<<metPhi<<" "
            << "Ht="<<Ht<<" "
            << "mHt="<<mHt<<" "
            << "mHtPhi="<<mHtPhi<<" "
            << "sumEt="<<sumEt<<std::endl;
}


#endif


