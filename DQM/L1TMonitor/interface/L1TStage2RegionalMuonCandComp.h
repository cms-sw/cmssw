#ifndef DQM_L1TMonitor_L1TStage2RegionalMuonCandComp_h
#define DQM_L1TMonitor_L1TStage2RegionalMuonCandComp_h


#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

namespace regionalmuoncandcompdqm {
  struct Histograms {
    ConcurrentMonitorElement summary;
    ConcurrentMonitorElement errorSummaryNum;
    ConcurrentMonitorElement errorSummaryDen;

    ConcurrentMonitorElement muColl1BxRange;
    ConcurrentMonitorElement muColl1nMu;
    ConcurrentMonitorElement muColl1hwPt;
    ConcurrentMonitorElement muColl1hwEta;
    ConcurrentMonitorElement muColl1hwPhi;
    ConcurrentMonitorElement muColl1hwSign;
    ConcurrentMonitorElement muColl1hwSignValid;
    ConcurrentMonitorElement muColl1hwQual;
    ConcurrentMonitorElement muColl1link;
    ConcurrentMonitorElement muColl1processor;
    ConcurrentMonitorElement muColl1trackFinderType;
    ConcurrentMonitorElement muColl1hwHF;
    ConcurrentMonitorElement muColl1TrkAddrSize;
    ConcurrentMonitorElement muColl1TrkAddr;

    ConcurrentMonitorElement muColl2BxRange;
    ConcurrentMonitorElement muColl2nMu;
    ConcurrentMonitorElement muColl2hwPt;
    ConcurrentMonitorElement muColl2hwEta;
    ConcurrentMonitorElement muColl2hwPhi;
    ConcurrentMonitorElement muColl2hwSign;
    ConcurrentMonitorElement muColl2hwSignValid;
    ConcurrentMonitorElement muColl2hwQual;
    ConcurrentMonitorElement muColl2link;
    ConcurrentMonitorElement muColl2processor;
    ConcurrentMonitorElement muColl2trackFinderType;
    ConcurrentMonitorElement muColl2hwHF;
    ConcurrentMonitorElement muColl2TrkAddrSize;
    ConcurrentMonitorElement muColl2TrkAddr;
  };
}

class L1TStage2RegionalMuonCandComp : public DQMGlobalEDAnalyzer<regionalmuoncandcompdqm::Histograms> {

 public:

  L1TStage2RegionalMuonCandComp(const edm::ParameterSet& ps);
  ~L1TStage2RegionalMuonCandComp() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

 protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&, regionalmuoncandcompdqm::Histograms&) const override;
  void bookHistograms(DQMStore::ConcurrentBooker&, const edm::Run&, const edm::EventSetup&, regionalmuoncandcompdqm::Histograms&) const override;
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, regionalmuoncandcompdqm::Histograms const&) const override;

 private:  

  enum variables {BXRANGEGOOD=1, BXRANGEBAD, NMUONGOOD, NMUONBAD, MUONALL, MUONGOOD, PTBAD, ETABAD, LOCALPHIBAD, SIGNBAD, SIGNVALBAD, QUALBAD, HFBAD, LINKBAD, PROCBAD, TFBAD, TRACKADDRBAD};
  enum ratioVariables {RBXRANGE=1, RNMUON, RMUON, RPT, RETA, RLOCALPHI, RSIGN, RSIGNVAL, RQUAL, RHF, RLINK, RPROC, RTF, RTRACKADDR};
  enum tfs {BMTFBIN=1, OMTFNEGBIN, OMTFPOSBIN, EMTFNEGBIN, EMTFPOSBIN};
  bool incBin[RTRACKADDR+1];

  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> muonToken1;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> muonToken2;
  std::string monitorDir;
  std::string muonColl1Title;
  std::string muonColl2Title;
  std::string summaryTitle;
  bool ignoreBadTrkAddr;
  std::vector<int> ignoreBin;
  bool verbose;
};

#endif
