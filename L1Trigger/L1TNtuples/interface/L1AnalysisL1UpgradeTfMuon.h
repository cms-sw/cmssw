#ifndef __L1Analysis_L1AnalysisL1UpgradeTfMuon_H__
#define __L1Analysis_L1AnalysisL1UpgradeTfMuon_H__

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

#include "L1AnalysisL1UpgradeTfMuonDataFormat.h"

namespace L1Analysis
{
  class L1AnalysisL1UpgradeTfMuon 
  {
  public:
    enum {TEST=0};
    L1AnalysisL1UpgradeTfMuon();
    ~L1AnalysisL1UpgradeTfMuon();
    void Reset() {l1upgradetfmuon_.Reset();}
    void SetTfMuon (const l1t::RegionalMuonCandBxCollection& muon, unsigned maxL1UpgradeTfMuon);
    L1AnalysisL1UpgradeTfMuonDataFormat * getData() {return &l1upgradetfmuon_;}

  private :
    L1AnalysisL1UpgradeTfMuonDataFormat l1upgradetfmuon_;
  }; 
}
#endif


