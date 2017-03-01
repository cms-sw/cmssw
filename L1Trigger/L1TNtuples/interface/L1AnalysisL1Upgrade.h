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

#include "L1AnalysisL1UpgradeDataFormat.h"

#include "L1Trigger/L1TMuon/interface/MicroGMTConfiguration.h"

namespace L1Analysis
{
  class L1AnalysisL1Upgrade 
  {
  public:
    enum {TEST=0};
    L1AnalysisL1Upgrade();
    ~L1AnalysisL1Upgrade();
    void Reset() {l1upgrade_.Reset();}
    void SetEm   (const edm::Handle<l1t::EGammaBxCollection> em,   unsigned maxL1Upgrade);
    void SetTau  (const edm::Handle<l1t::TauBxCollection>    tau,  unsigned maxL1Upgrade);
    void SetJet  (const edm::Handle<l1t::JetBxCollection>    jet,  unsigned maxL1Upgrade);
    void SetSum  (const edm::Handle<l1t::EtSumBxCollection>  sums, unsigned maxL1Upgrade);
    void SetMuon (const edm::Handle<l1t::MuonBxCollection>   muon, unsigned maxL1Upgrade);
    L1AnalysisL1UpgradeDataFormat * getData() {return &l1upgrade_;}

  private :
    L1AnalysisL1UpgradeDataFormat l1upgrade_;
  }; 
}
#endif


