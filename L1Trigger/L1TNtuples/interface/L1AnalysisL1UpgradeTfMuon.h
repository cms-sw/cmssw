#ifndef __L1Analysis_L1AnalysisL1UpgradeTfMuon_H__
#define __L1Analysis_L1AnalysisL1UpgradeTfMuon_H__

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeTfMuonDataFormat.h"
#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"

namespace L1Analysis {
  class L1AnalysisL1UpgradeTfMuon {
  public:
    enum { TEST = 0 };
    L1AnalysisL1UpgradeTfMuon();
    ~L1AnalysisL1UpgradeTfMuon();
    void Reset() {
      l1upgradetfmuon_.Reset();
      isRun3_ = false;
    }
    void SetRun3Muons() { isRun3_ = true; }
    void SetTfMuon(const l1t::RegionalMuonCandBxCollection& muon, unsigned maxL1UpgradeTfMuon);
    L1AnalysisL1UpgradeTfMuonDataFormat* getData() { return &l1upgradetfmuon_; }

  private:
    L1AnalysisL1UpgradeTfMuonDataFormat l1upgradetfmuon_;
    bool isRun3_{false};
  };
}  // namespace L1Analysis
#endif
