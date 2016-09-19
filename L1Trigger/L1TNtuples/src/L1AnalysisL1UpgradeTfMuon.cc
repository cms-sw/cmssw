#include "L1Trigger/L1TNtuples/interface/L1AnalysisL1UpgradeTfMuon.h"
#include <cmath>
L1Analysis::L1AnalysisL1UpgradeTfMuon::L1AnalysisL1UpgradeTfMuon()
{
}

L1Analysis::L1AnalysisL1UpgradeTfMuon::~L1AnalysisL1UpgradeTfMuon()
{

}

void L1Analysis::L1AnalysisL1UpgradeTfMuon::SetTfMuon(const l1t::RegionalMuonCandBxCollection& muon, unsigned maxL1UpgradeTfMuon)
{
  for (int ibx = muon.getFirstBX(); ibx <= muon.getLastBX(); ++ibx) {
    for (l1t::RegionalMuonCandBxCollection::const_iterator it = muon.begin(ibx); it != muon.end(ibx) && l1upgradetfmuon_.nTfMuons < maxL1UpgradeTfMuon; ++it){
      if (it->hwPt() > 0) {
	
        l1upgradetfmuon_.tfMuonHwPt.push_back(it->hwPt());
        l1upgradetfmuon_.tfMuonHwEta.push_back(it->hwEta());
        l1upgradetfmuon_.tfMuonHwPhi.push_back(it->hwPhi());
	l1upgradetfmuon_.tfMuonGlobalPhi.push_back(l1t::MicroGMTConfiguration::calcGlobalPhi(it->hwPhi(),it->trackFinderType(),it->processor()));	
        l1upgradetfmuon_.tfMuonHwSign.push_back(it->hwSign());
        l1upgradetfmuon_.tfMuonHwSignValid.push_back(it->hwSignValid());
        l1upgradetfmuon_.tfMuonHwQual.push_back(it->hwQual());
        l1upgradetfmuon_.tfMuonLink.push_back(it->link());
        l1upgradetfmuon_.tfMuonProcessor.push_back(it->processor());
        l1upgradetfmuon_.tfMuonTrackFinderType.push_back(it->trackFinderType());
        l1upgradetfmuon_.tfMuonHwHF.push_back(it->hwHF());
        l1upgradetfmuon_.tfMuonBx.push_back(ibx);
        std::map<int, int>  trAdd;
        trAdd = it->trackAddress();
        int wheel = pow(-1,trAdd[0]) * trAdd[1];
        l1upgradetfmuon_.tfMuonWh.push_back(wheel);
        l1upgradetfmuon_.tfMuonTrAdd.push_back(trAdd[2]);
        l1upgradetfmuon_.tfMuonTrAdd.push_back(trAdd[3]);
        l1upgradetfmuon_.tfMuonTrAdd.push_back(trAdd[4]);
        l1upgradetfmuon_.tfMuonTrAdd.push_back(trAdd[5]);


        l1upgradetfmuon_.nTfMuons++;
      }
    }
  }
}

