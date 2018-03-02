#ifndef DQM_L1TMonitor_L1TdeStage2EMTF_h
#define DQM_L1TMonitor_L1TdeStage2EMTF_h

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "DQMServices/Core/interface/DQMGlobalEDAnalyzer.h"
#include "DQMServices/Core/interface/ConcurrentMonitorElement.h"

#include "DataFormats/L1TMuon/interface/RegionalMuonCand.h"

namespace emtfdedqm {
  struct Histograms {
    ConcurrentMonitorElement emtfComparenMuonsEvent;

    ConcurrentMonitorElement emtfDataBX;
    ConcurrentMonitorElement emtfEmulBX;
    ConcurrentMonitorElement emtfDatahwPt;
    ConcurrentMonitorElement emtfEmulhwPt;
    ConcurrentMonitorElement emtfDatahwEta;
    ConcurrentMonitorElement emtfEmulhwEta;
    ConcurrentMonitorElement emtfDatahwPhi;
    ConcurrentMonitorElement emtfEmulhwPhi;
    ConcurrentMonitorElement emtfDatahwQual;
    ConcurrentMonitorElement emtfEmulhwQual;

    /*ConcurrentMonitorElement emtfComparehwPt;
    ConcurrentMonitorElement emtfComparehwEta;
    ConcurrentMonitorElement emtfComparehwPhi;
    ConcurrentMonitorElement emtfComparehwQual;*/
  };
}

class L1TdeStage2EMTF : public DQMGlobalEDAnalyzer<emtfdedqm::Histograms> {

 public:

  L1TdeStage2EMTF(const edm::ParameterSet& ps);
  ~L1TdeStage2EMTF() override;

 protected:

  void dqmBeginRun(const edm::Run&, const edm::EventSetup&, emtfdedqm::Histograms&) const override;
  void bookHistograms(DQMStore::ConcurrentBooker&, const edm::Run&, const edm::EventSetup&, emtfdedqm::Histograms&) const override;
  void dqmAnalyze(const edm::Event&, const edm::EventSetup&, const emtfdedqm::Histograms&) const override;

 private:

  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> dataToken;
  edm::EDGetTokenT<l1t::RegionalMuonCandBxCollection> emulToken;
  std::string monitorDir;
  bool verbose;
};

#endif
