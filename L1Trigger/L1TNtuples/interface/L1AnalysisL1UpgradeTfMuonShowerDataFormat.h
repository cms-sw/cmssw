#ifndef __L1Analysis_L1AnalysisL1UpgradeTfMuonShowerDataFormat_H__
#define __L1Analysis_L1AnalysisL1UpgradeTfMuonShowerDataFormat_H__

#include <vector>
#include <map>

namespace L1Analysis {

  struct L1AnalysisL1UpgradeTfMuonShowerDataFormat {
    L1AnalysisL1UpgradeTfMuonShowerDataFormat() { Reset(); };
    ~L1AnalysisL1UpgradeTfMuonShowerDataFormat(){};

    void Reset() {
      nTfMuonShowers = 0;
      tfMuonShowerBx.clear();
      tfMuonShowerOneNominal.clear();
      tfMuonShowerOneTight.clear();
      tfMuonShowerOneLoose.clear();
      tfMuonShowerTwoLoose.clear();
      tfMuonShowerEndcap.clear();
      tfMuonShowerSector.clear();
    }

    unsigned short int nTfMuonShowers;
    std::vector<short int> tfMuonShowerBx;
    std::vector<short int> tfMuonShowerOneNominal;
    std::vector<short int> tfMuonShowerOneTight;
    std::vector<short int> tfMuonShowerOneLoose;
    std::vector<short int> tfMuonShowerTwoLoose;
    std::vector<short int> tfMuonShowerEndcap;
    std::vector<short int> tfMuonShowerSector;
  };
}  // namespace L1Analysis
#endif
