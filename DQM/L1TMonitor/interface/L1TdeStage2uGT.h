#ifndef L1TdeStage2uGT_H
#define L1TdeStage2uGT_H

// system include files
#include <memory>
#include <string>
#include <array>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"

// #include "FWCore/ServiceRegistry/interface/Service.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "FWCore/Framework/interface/ESHandle.h"
#include "FWCore/Framework/interface/LuminosityBlock.h"

#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"

#include "DataFormats/L1TGlobal/interface/GlobalAlgBlk.h"
// #include "DataFormats/Provenance/interface/EventAuxiliary.h"

#include "CondFormats/RunInfo/interface/RunInfo.h"
#include "CondFormats/DataRecord/interface/RunSummaryRcd.h"

#include "L1Trigger/L1TGlobal/interface/L1TGlobalUtil.h"

#include "FWCore/Utilities/interface/RegexMatch.h"

class L1TdeStage2uGT : public DQMEDAnalyzer {
public:
  L1TdeStage2uGT(const edm::ParameterSet& ps);
  ~L1TdeStage2uGT() override;

protected:
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
  void bookHistograms(DQMStore::IBooker& ibooker, const edm::Run&, const edm::EventSetup&) override;

private:
  // Input and config info
  edm::InputTag dataLabel_;
  edm::EDGetTokenT<GlobalAlgBlkBxCollection> dataSource_;
  edm::InputTag emulLabel_;
  edm::EDGetTokenT<GlobalAlgBlkBxCollection> emulSource_;
  std::vector<std::string> triggerBlackList_;
  int numBx_;
  std::string histFolder_;
  l1t::L1TGlobalUtil gtUtil_;
  int numLS_;
  uint m_currentLumi;
  uint m_currentRun;

  int firstBx, lastBx;

  std::map<std::string, MonitorElement*> m_HistNamesInitial, m_HistNamesFinal, m_SummaryHistograms;
  MonitorElement* initDecisionMismatches_vs_LS;
  MonitorElement* finalDecisionMismatches_vs_LS;
  MonitorElement* m_normalizationHisto;

  enum SummaryColumn {
    NInitalMismatchDataNoEmul,
    NInitalMismatchEmulNoData,
    NFinalMismatchDataNoEmul,
    NFinalMismatchEmulNoData,
    NSummaryColumns,
  };

  void fillHist(const std::map<std::string, MonitorElement*>&, const std::string&, const Double_t&, const Double_t&);
};

#endif
