#ifndef __L1Analysis_L1AnalysisL1UpgradeTfMuonDataFormat_H__
#define __L1Analysis_L1AnalysisL1UpgradeTfMuonDataFormat_H__

#include <vector>

namespace L1Analysis
{
  struct L1AnalysisL1UpgradeTfMuonDataFormat
  {
    L1AnalysisL1UpgradeTfMuonDataFormat(){ Reset();};
    ~L1AnalysisL1UpgradeTfMuonDataFormat(){};
    
    void Reset()
    {
      nTfMuons = 0;
      tfMuonHwPt.clear();
      tfMuonHwEta.clear();
      tfMuonHwPhi.clear();
      tfMuonHwSign.clear();
      tfMuonHwSignValid.clear();
      tfMuonHwQual.clear();
      tfMuonLink.clear();
      tfMuonProcessor.clear();
      tfMuonTrackFinderType.clear();
      tfMuonHwHF.clear();
      tfMuonBx.clear();
    }
   
    unsigned short int nTfMuons;
    std::vector<short int> tfMuonHwPt;
    std::vector<short int> tfMuonHwEta;
    std::vector<short int> tfMuonHwPhi;
    std::vector<short int> tfMuonHwSign;
    std::vector<short int> tfMuonHwSignValid;
    std::vector<short int> tfMuonHwQual;
    std::vector<short int> tfMuonLink;
    std::vector<short int> tfMuonProcessor;
    std::vector<short int> tfMuonTrackFinderType;
    std::vector<short int> tfMuonHwHF;
    std::vector<short int> tfMuonBx;
  }; 
}
#endif

