#ifndef __L1Analysis_L1AnalysisL1UpgradeTfMuonDataFormat_H__
#define __L1Analysis_L1AnalysisL1UpgradeTfMuonDataFormat_H__

#include "L1Trigger/L1TMuon/interface/RegionalMuonRawDigiTranslator.h"
#include <vector>
#include <map>

namespace L1Analysis {
  struct L1AnalysisL1UpgradeTfMuonDataFormat {
    L1AnalysisL1UpgradeTfMuonDataFormat() { Reset(); };
    ~L1AnalysisL1UpgradeTfMuonDataFormat(){};

    void Reset() {
      nTfMuons = 0;
      tfMuonHwPt.clear();
      tfMuonHwEta.clear();
      tfMuonHwPhi.clear();
      tfMuonGlobalPhi.clear();
      tfMuonHwSign.clear();
      tfMuonHwSignValid.clear();
      tfMuonHwQual.clear();
      tfMuonLink.clear();
      tfMuonProcessor.clear();
      tfMuonTrackFinderType.clear();
      tfMuonHwHF.clear();
      tfMuonBx.clear();
      tfMuonWh.clear();
      tfMuonTrAdd.clear();
      tfMuonDecodedTrAdd.clear();
      tfMuonHwTrAdd.clear();
    }

    unsigned short int nTfMuons;
    std::vector<short int> tfMuonHwPt;
    std::vector<short int> tfMuonHwEta;
    std::vector<short int> tfMuonHwPhi;
    std::vector<short int> tfMuonGlobalPhi;
    std::vector<short int> tfMuonHwSign;
    std::vector<short int> tfMuonHwSignValid;
    std::vector<short int> tfMuonHwQual;
    std::vector<short int> tfMuonLink;
    std::vector<short int> tfMuonProcessor;
    std::vector<short int> tfMuonTrackFinderType;
    std::vector<short int> tfMuonHwHF;
    std::vector<short int> tfMuonBx;
    std::vector<short int> tfMuonWh;
    std::vector<short int> tfMuonTrAdd;
    std::vector<std::map<std::string, int>> tfMuonDecodedTrAdd;
    std::vector<short int> tfMuonHwTrAdd;
  };
}  // namespace L1Analysis
#endif
