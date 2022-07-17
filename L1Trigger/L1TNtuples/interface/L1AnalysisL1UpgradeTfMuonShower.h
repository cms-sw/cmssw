#ifndef __L1Analysis_L1AnalysisL1UpgradeTfMuonShower_H__
#define __L1Analysis_L1AnalysisL1UpgradeTfMuonShower_H__

#include "DataFormats/L1TMuon/interface/RegionalMuonShower.h"

#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeTfMuonShowerDataFormat.h"
namespace L1Analysis {
  class L1AnalysisL1UpgradeTfMuonShower {
  public:
    enum { TEST = 0 };
    L1AnalysisL1UpgradeTfMuonShower();
    ~L1AnalysisL1UpgradeTfMuonShower();
    void Reset() { l1upgradetfmuonshower_.Reset(); }
    void SetTfMuonShower(const l1t::RegionalMuonShowerBxCollection& muon, unsigned maxL1UpgradeTfMuonShower);
    L1AnalysisL1UpgradeTfMuonShowerDataFormat* getData() { return &l1upgradetfmuonshower_; }

  private:
    L1AnalysisL1UpgradeTfMuonShowerDataFormat l1upgradetfmuonshower_;
  };
}  // namespace L1Analysis
#endif
