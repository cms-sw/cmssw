#ifndef EcalMEFormatter_H
#define EcalMEFormatter_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "DQM/EcalCommon/interface/DQWorker.h"
#include "DQM/EcalCommon/interface/MESet.h"

class EcalMEFormatter : public edm::EDAnalyzer, public ecaldqm::DQWorker {
 public:
  EcalMEFormatter(edm::ParameterSet const&);
  ~EcalMEFormatter() {};

  static void fillDescriptions(edm::ConfigurationDescriptions&);

 private:
  void endLuminosityBlock(edm::LuminosityBlock const&, edm::EventSetup const&) override;
  void endRun(edm::Run const&, edm::EventSetup const&) override;
  void analyze(edm::Event const&, edm::EventSetup const&) override {}

  void format_(bool);
  void formatDet2D_(ecaldqm::MESet&);
};

#endif
