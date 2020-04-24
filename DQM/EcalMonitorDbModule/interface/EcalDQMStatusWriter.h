#ifndef EcalDQMStatusWriter_H
#define EcalDQMStatusWriter_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"

class EcalDQMStatusWriter : public edm::EDAnalyzer {
 public:
  EcalDQMStatusWriter(edm::ParameterSet const&);
  ~EcalDQMStatusWriter() {}

 private:
  void analyze(edm::Event const&, edm::EventSetup const&) override;

  EcalDQMChannelStatus channelStatus_;
  EcalDQMTowerStatus towerStatus_;
  unsigned firstRun_;
};

#endif
