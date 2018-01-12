#ifndef DQM_L1TMonitor_L1TStage2MuonComp_h
#define DQM_L1TMonitor_L1TStage2MuonComp_h


#include "DataFormats/L1Trigger/interface/Muon.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace muoncompdqm {
  struct Histograms {
    ConcurrentMonitorElement summary;
    ConcurrentMonitorElement errorSummaryNum;
    ConcurrentMonitorElement errorSummaryDen;

    ConcurrentMonitorElement muColl1BxRange;
    ConcurrentMonitorElement muColl1nMu;
    ConcurrentMonitorElement muColl1hwPt;
    ConcurrentMonitorElement muColl1hwEta;
    ConcurrentMonitorElement muColl1hwPhi;
    ConcurrentMonitorElement muColl1hwEtaAtVtx;
    ConcurrentMonitorElement muColl1hwPhiAtVtx;
    ConcurrentMonitorElement muColl1hwCharge;
    ConcurrentMonitorElement muColl1hwChargeValid;
    ConcurrentMonitorElement muColl1hwQual;
    ConcurrentMonitorElement muColl1hwIso;
    ConcurrentMonitorElement muColl1Index;

    ConcurrentMonitorElement muColl2BxRange;
    ConcurrentMonitorElement muColl2nMu;
    ConcurrentMonitorElement muColl2hwPt;
    ConcurrentMonitorElement muColl2hwEta;
    ConcurrentMonitorElement muColl2hwPhi;
    ConcurrentMonitorElement muColl2hwEtaAtVtx;
    ConcurrentMonitorElement muColl2hwPhiAtVtx;
    ConcurrentMonitorElement muColl2hwCharge;
    ConcurrentMonitorElement muColl2hwChargeValid;
    ConcurrentMonitorElement muColl2hwQual;
    ConcurrentMonitorElement muColl2hwIso;
    ConcurrentMonitorElement muColl2Index;
  };
}

class L1TStage2MuonComp : public DQMGlobalEDAnalyzer<muoncompdqm::Histograms> {

 public:

  L1TStage2MuonComp(const edm::ParameterSet& ps);
  ~L1TStage2MuonComp() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&, muoncompdqm::Histograms &) const override;
  void bookHistograms(DQMStore::ConcurrentBooker&, const edm::Run&, const edm::EventSetup&, muoncompdqm::Histograms &) const override;
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, muoncompdqm::Histograms const&) const override;

 private:  

  enum variables {BXRANGEGOOD=1, BXRANGEBAD, NMUONGOOD, NMUONBAD, MUONALL, MUONGOOD, PTBAD, ETABAD, PHIBAD, ETAATVTXBAD, PHIATVTXBAD, CHARGEBAD, CHARGEVALBAD, QUALBAD, ISOBAD, IDXBAD};
  enum ratioVariables {RBXRANGE=1, RNMUON, RMUON, RPT, RETA, RPHI, RETAATVTX, RPHIATVTX, RCHARGE, RCHARGEVAL, RQUAL, RISO, RIDX};
  bool incBin[RIDX+1];

  edm::EDGetTokenT<l1t::MuonBxCollection> muonToken1;
  edm::EDGetTokenT<l1t::MuonBxCollection> muonToken2;
  std::string monitorDir;
  std::string muonColl1Title;
  std::string muonColl2Title;
  std::string summaryTitle;
  std::vector<int> ignoreBin;
  bool verbose;

};

#endif
