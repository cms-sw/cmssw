#ifndef DQM_L1TMonitor_L1TStage2RegionalMuonCandComp_h
#define DQM_L1TMonitor_L1TStage2RegionalMuonCandComp_h

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class L1TStage2RegionalMuonCandComp : public DQMEDAnalyzer {
public:
  L1TStage2RegionalMuonCandComp(const edm::ParameterSet& ps);
  ~L1TStage2RegionalMuonCandComp() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  enum variables {
    BXRANGEGOOD = 1,
    BXRANGEBAD,
    NMUONGOOD,
    NMUONBAD,
    MUONALL,
    MUONGOOD,
    PTBAD,
    ETABAD,
    LOCALPHIBAD,
    SIGNBAD,
    SIGNVALBAD,
    QUALBAD,
    HFBAD,
    LINKBAD,
    PROCBAD,
    TFBAD,
    TRACKADDRBAD,
    DXYBAD,
    PT2BAD
  };
  enum ratioVariables {
    RBXRANGE = 1,
    RNMUON,
    RMUON,
    RPT,
    RETA,
    RLOCALPHI,
    RSIGN,
    RSIGNVAL,
    RQUAL,
    RHF,
    RLINK,
    RPROC,
    RTF,
    RTRACKADDR,
    RDXY,
    RPT2
  };
  enum tfs { BMTFBIN = 1, OMTFNEGBIN, OMTFPOSBIN, EMTFNEGBIN, EMTFPOSBIN };
  bool incBin[RPT2 + 1];

  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> muonToken1;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> muonToken2;
  std::string monitorDir;
  std::string muonColl1Title;
  std::string muonColl2Title;
  std::string summaryTitle;
  bool ignoreBadTrkAddr;
  std::vector<int> ignoreBin;
  bool verbose;
  bool isBmtf;

  MonitorElement* summary;
  MonitorElement* errorSummaryNum;
  MonitorElement* errorSummaryDen;

  MonitorElement* muColl1BxRange;
  MonitorElement* muColl1nMu;
  MonitorElement* muColl1hwPt;
  MonitorElement* muColl1hwEta;
  MonitorElement* muColl1hwPhi;
  MonitorElement* muColl1hwSign;
  MonitorElement* muColl1hwSignValid;
  MonitorElement* muColl1hwQual;
  MonitorElement* muColl1link;
  MonitorElement* muColl1processor;
  MonitorElement* muColl1trackFinderType;
  MonitorElement* muColl1hwHF;
  MonitorElement* muColl1TrkAddrSize;
  MonitorElement* muColl1TrkAddr;
  MonitorElement* muColl1hwDXY;
  MonitorElement* muColl1hwPt2;

  MonitorElement* muColl2BxRange;
  MonitorElement* muColl2nMu;
  MonitorElement* muColl2hwPt;
  MonitorElement* muColl2hwEta;
  MonitorElement* muColl2hwPhi;
  MonitorElement* muColl2hwSign;
  MonitorElement* muColl2hwSignValid;
  MonitorElement* muColl2hwQual;
  MonitorElement* muColl2link;
  MonitorElement* muColl2processor;
  MonitorElement* muColl2trackFinderType;
  MonitorElement* muColl2hwHF;
  MonitorElement* muColl2TrkAddrSize;
  MonitorElement* muColl2TrkAddr;
  MonitorElement* muColl2hwDXY;
  MonitorElement* muColl2hwPt2;
};

#endif
