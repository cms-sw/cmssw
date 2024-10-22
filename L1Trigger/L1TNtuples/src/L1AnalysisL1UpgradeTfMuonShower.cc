#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeTfMuonShower.h"
#include <cmath>
L1Analysis::L1AnalysisL1UpgradeTfMuonShower::L1AnalysisL1UpgradeTfMuonShower() {}

L1Analysis::L1AnalysisL1UpgradeTfMuonShower::~L1AnalysisL1UpgradeTfMuonShower() {}

void L1Analysis::L1AnalysisL1UpgradeTfMuonShower::SetTfMuonShower(const l1t::RegionalMuonShowerBxCollection& muonShower,
                                                                  unsigned maxL1UpgradeTfMuonShower) {
  for (int ibx = muonShower.getFirstBX(); ibx <= muonShower.getLastBX(); ++ibx) {
    for (auto it = muonShower.begin(ibx);
         it != muonShower.end(ibx) && l1upgradetfmuonshower_.nTfMuonShowers < maxL1UpgradeTfMuonShower;
         ++it) {
      if (it->isValid()) {
        l1upgradetfmuonshower_.tfMuonShowerBx.push_back(ibx);
        l1upgradetfmuonshower_.tfMuonShowerEndcap.push_back(it->trackFinderType() == l1t::tftype::emtf_pos ? 1 : -1);
        l1upgradetfmuonshower_.tfMuonShowerSector.push_back(it->processor() + 1);
        l1upgradetfmuonshower_.tfMuonShowerOneNominal.push_back(it->isOneNominalInTime());
        l1upgradetfmuonshower_.tfMuonShowerOneTight.push_back(it->isOneTightInTime());
        l1upgradetfmuonshower_.tfMuonShowerOneLoose.push_back(it->isOneLooseInTime());
        l1upgradetfmuonshower_.tfMuonShowerTwoLoose.push_back(it->isTwoLooseInTime());
        l1upgradetfmuonshower_.nTfMuonShowers++;
      }
    }
  }
}
