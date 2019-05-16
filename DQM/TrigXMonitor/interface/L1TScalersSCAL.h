#ifndef L1TScalersSCAL_H
#define L1TScalersSCAL_H

#include <vector>

#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/interface/DQMEDAnalyzer.h"
#include "FWCore/Utilities/interface/InputTag.h"

#include "DataFormats/Scalers/interface/Level1TriggerScalers.h"
#include "DataFormats/Scalers/interface/LumiScalers.h"
#include "DataFormats/Scalers/interface/L1AcceptBunchCrossing.h"

class L1TScalersSCAL : public DQMEDAnalyzer {
 public:
  enum { N_LUMISECTION_TIME = 93 };
  
  L1TScalersSCAL(const edm::ParameterSet& ps);
  ~L1TScalersSCAL() override;
  void bookHistograms(DQMStore::IBooker &, edm::Run const &,
                      edm::EventSetup const &) override;
  void analyze(const edm::Event& e, const edm::EventSetup& c) override;
 private:
  edm::EDGetTokenT<Level1TriggerScalersCollection> l1triggerscalers_;
  edm::EDGetTokenT<LumiScalersCollection> lumiscalers_;
  edm::EDGetTokenT<L1AcceptBunchCrossingCollection> l1acceptBX_;

  bool verbose_, denomIsTech_, monitorDaemon_;
  unsigned int denomBit_, muonBit_, egammaBit_, jetBit_;
  int nev_;  // Number of events processed
  long reftime_, buffertime_;
  std::vector<double> algorithmRates_;
  std::vector<double> bufferAlgoRates_;
  std::vector<double> technicalRates_;
  std::vector<double> bufferTechRates_;
  std::vector<double> integral_algo_;
  std::vector<double> integral_tech_;
  double integral_tech_42_OR_43_;
  unsigned int bufferLumi_;

  MonitorElement* orbitNum;
  MonitorElement* trigNum;
  MonitorElement* eventNum;
  MonitorElement* physTrig;
  MonitorElement* randTrig;
  MonitorElement* numberResets;
  MonitorElement* deadTime;
  MonitorElement* lostFinalTriggers;
  MonitorElement* algoRate[128];
  MonitorElement* techRate[64];
  MonitorElement* integralAlgo[128];
  MonitorElement* integralTech[64];
  MonitorElement* integralTech_42_OR_43;
  MonitorElement* techRateRatio_33_over_32;
  MonitorElement* techRateRatio_8;
  MonitorElement* techRateRatio_9;
  MonitorElement* techRateRatio_10;
  MonitorElement* techRateRatio_36;
  MonitorElement* techRateRatio_37;
  MonitorElement* techRateRatio_38;
  MonitorElement* techRateRatio_39;
  MonitorElement* techRateRatio_40;
  MonitorElement* techRateRatio_41;
  MonitorElement* techRateRatio_42;
  MonitorElement* techRateRatio_43;
  MonitorElement* rateRatio_mu;
  MonitorElement* rateRatio_egamma;
  MonitorElement* rateRatio_jet;

  MonitorElement* physRate;
  MonitorElement* randRate;
  MonitorElement* deadTimePercent;
  MonitorElement* lostPhysRate;
  MonitorElement* lostPhysRateBeamActive;
  MonitorElement* instTrigRate;
  MonitorElement* instEventRate;

  MonitorElement* instLumi;
  MonitorElement* instLumiErr;
  MonitorElement* instLumiQlty;
  MonitorElement* instEtLumi;
  MonitorElement* instEtLumiErr;
  MonitorElement* instEtLumiQlty;
  MonitorElement* sectionNum;
  MonitorElement* startOrbit;
  MonitorElement* numOrbits;

  MonitorElement* orbitNumL1A[4];
  MonitorElement* bunchCrossingL1A[4];
  MonitorElement* bunchCrossingCorr[3];
  MonitorElement* bunchCrossingDiff[3];
  MonitorElement* bunchCrossingDiff_small[3];
};

#endif  // L1TScalersSCAL_H
