#ifndef __L1Analysis_L1AnalysisL1Upgrade_H__
#define __L1Analysis_L1AnalysisL1Upgrade_H__

//-------------------------------------------------------------------------------
// Created 02/03/2010 - A.C. Le Bihan
//
//
// Original code : L1TriggerDPG/L1Ntuples/L1UpgradeTreeProducer - Jim Brooke
//-------------------------------------------------------------------------------

#include "DataFormats/L1Trigger/interface/EGamma.h"
#include "DataFormats/L1Trigger/interface/Tau.h"
#include "DataFormats/L1Trigger/interface/Jet.h"
#include "DataFormats/L1Trigger/interface/Muon.h"
#include "DataFormats/L1Trigger/interface/EtSum.h"
#include "DataFormats/L1Trigger/interface/MuonShower.h"

#include "L1AnalysisL1UpgradeDataFormat.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"

namespace L1Analysis {
  class L1AnalysisL1Upgrade {
  public:
    enum { TEST = 0 };
    L1AnalysisL1Upgrade();
    ~L1AnalysisL1Upgrade();
    void Reset() { l1upgrade_.Reset(); }
    void SetEm(const edm::Handle<l1t::EGammaBxCollection> em, unsigned maxL1Upgrade) { SetEm(*em, maxL1Upgrade); }
    void SetTau(const edm::Handle<l1t::TauBxCollection> tau, unsigned maxL1Upgrade) { SetTau(*tau, maxL1Upgrade); }
    void SetJet(const edm::Handle<l1t::JetBxCollection> jet, unsigned maxL1Upgrade) { SetJet(*jet, maxL1Upgrade); }
    void SetSum(const edm::Handle<l1t::EtSumBxCollection> sums, unsigned maxL1Upgrade) { SetSum(*sums, maxL1Upgrade); }
    void SetMuon(const edm::Handle<l1t::MuonBxCollection> muon, unsigned maxL1Upgrade) { SetMuon(*muon, maxL1Upgrade); }
    void SetMuonShower(const edm::Handle<l1t::MuonShowerBxCollection> muonShower, unsigned maxL1Upgrade) {
      SetMuonShower(*muonShower, maxL1Upgrade);
    }
    void SetEm(const l1t::EGammaBxCollection& em, unsigned maxL1Upgrade);
    void SetTau(const l1t::TauBxCollection& tau, unsigned maxL1Upgrade);
    void SetJet(const l1t::JetBxCollection& jet, unsigned maxL1Upgrade);
    void SetSum(const l1t::EtSumBxCollection& sums, unsigned maxL1Upgrade);
    void SetMuon(const l1t::MuonBxCollection& muon, unsigned maxL1Upgrade);
    void SetMuonShower(const l1t::MuonShowerBxCollection& muonShower, unsigned maxL1Upgrade);

    L1AnalysisL1UpgradeDataFormat* getData() { return &l1upgrade_; }

  private:
    L1AnalysisL1UpgradeDataFormat l1upgrade_;
  };
}  // namespace L1Analysis
#endif
