#ifndef __L1Analysis_L1AnalysisRecoJet_H__
#define __L1Analysis_L1AnalysisRecoJet_H__

#include <TTree.h>
#include <iostream>
#include <vector>

namespace L1Analysis
{
  class L1AnalysisRecoJet
  {

  public :
    void initTree(TTree * tree, const std::string & className);

  public:
    L1AnalysisRecoJet() {}
    void print();

    // ---- General L1AnalysisRecoJet information.
    unsigned nJets;
    std::vector<double> e;
    std::vector<double> et;
    std::vector<double> etCorr;
    std::vector<double> corrFactor;
    std::vector<double> eta;
    std::vector<double> phi;
    std::vector<double> eEMF;
    std::vector<double> eHadHB;
    std::vector<double> eHadHE;
    std::vector<double> eHadHO;
    std::vector<double> eHadHF;
    std::vector<double> eEmEB;
    std::vector<double> eEmEE;
    std::vector<double> eEmHF;
    std::vector<double> eMaxEcalTow;
    std::vector<double> eMaxHcalTow;
    std::vector<double> towerArea;
    std::vector<int> towerSize;
    std::vector<int> n60;
    std::vector<int> n90;
};
}

void L1Analysis::L1AnalysisRecoJet::initTree(TTree * tree, const std::string & className)
{
     SetBranchAddress(tree,"nJets", className,       &nJets      );
     SetBranchAddress(tree,"e", className,           &e          );
     SetBranchAddress(tree,"et", className,          &et         );
     SetBranchAddress(tree,"etCorr", className,      &etCorr     );
     SetBranchAddress(tree,"corrFactor", className,  &corrFactor );
     SetBranchAddress(tree,"eta", className,         &eta        );
     SetBranchAddress(tree,"phi", className,         &phi        );
     SetBranchAddress(tree,"eEMF", className,        &eEMF       );
     SetBranchAddress(tree,"eHadHB", className,      &eHadHB     );
     SetBranchAddress(tree,"eHadHE", className,      &eHadHE     );
     SetBranchAddress(tree,"eHadHO", className,      &eHadHO     );
     SetBranchAddress(tree,"eHadHF", className,      &eHadHF     );
     SetBranchAddress(tree,"eEmEB", className,       &eEmEB      );
     SetBranchAddress(tree,"eEmEE", className,       &eEmEE      );
     SetBranchAddress(tree,"eEmHF", className,       &eEmHF     );
     SetBranchAddress(tree,"eMaxEcalTow", className, &eMaxEcalTow);
     SetBranchAddress(tree,"eMaxHcalTow", className, &eMaxHcalTow);
     SetBranchAddress(tree,"towerArea", className,   &towerArea  );
     SetBranchAddress(tree,"towerSize", className,   &towerSize  );
     SetBranchAddress(tree,"n60", className,         &n60        );
     SetBranchAddress(tree,"n90", className,         &n90        );
}

void L1Analysis::L1AnalysisRecoJet::print()
{
}



#endif


