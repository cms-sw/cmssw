#ifndef DQM_L1TMonitor_L1TStage2MuonShowerComp_h
#define DQM_L1TMonitor_L1TStage2MuonShowerComp_h

#include "DataFormats/L1Trigger/interface/MuonShower.h"

#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "DQMServices/Core/interface/DQMStore.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"

class L1TStage2MuonShowerComp : public DQMEDAnalyzer {
public:
  L1TStage2MuonShowerComp(const edm::ParameterSet& ps);
  ~L1TStage2MuonShowerComp() override;
  static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);

protected:
  void bookHistograms(DQMStore::IBooker&, const edm::Run&, const edm::EventSetup&) override;
  void analyze(const edm::Event&, const edm::EventSetup&) override;

private:
  enum variables {
    BXRANGEGOOD = 1,
    BXRANGEBAD,
    NSHOWERGOOD,
    NSHOWERBAD,
    SHOWERALL,
    SHOWERGOOD,
    NOMINALBAD,
    TIGHTBAD,
    LOOSEBAD
  };
  enum ratioVariables { RBXRANGE = 1, RNSHOWER, RSHOWER, RNOMINAL, RTIGHT, RLOOSE };
  int numSummaryBins_{LOOSEBAD};
  int numErrBins_{RLOOSE};
  bool incBin_[RLOOSE + 1];

  edm::EDGetTokenT<l1t::MuonShowerBxCollection> showerToken1_;
  edm::EDGetTokenT<l1t::MuonShowerBxCollection> showerToken2_;
  std::string monitorDir_;
  std::string showerColl1Title_;
  std::string showerColl2Title_;
  std::string summaryTitle_;
  std::vector<int> ignoreBin_;
  bool verbose_;

  MonitorElement* summary_;
  MonitorElement* errorSummaryNum_;
  MonitorElement* errorSummaryDen_;

  MonitorElement* showerColl1BxRange_;
  MonitorElement* showerColl1nShowers_;
  MonitorElement* showerColl1ShowerTypeVsBX_;

  MonitorElement* showerColl2BxRange_;
  MonitorElement* showerColl2nShowers_;
  MonitorElement* showerColl2ShowerTypeVsBX_;

  static constexpr unsigned IDX_LOOSE_SHOWER{3};
  static constexpr unsigned IDX_TIGHT_SHOWER{2};
  static constexpr unsigned IDX_NOMINAL_SHOWER{1};
};

#endif
