#ifndef EcalDQMStatusWriter_H
#define EcalDQMStatusWriter_H

#include "FWCore/Framework/interface/EDAnalyzer.h"

#include "CondFormats/EcalObjects/interface/EcalDQMChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalDQMTowerStatus.h"

#include "Geometry/EcalMapping/interface/EcalElectronicsMapping.h"
#include "Geometry/EcalMapping/interface/EcalMappingRcd.h"

#include <fstream>

class EcalDQMStatusWriter : public edm::EDAnalyzer {
public:
  EcalDQMStatusWriter(edm::ParameterSet const &);
  ~EcalDQMStatusWriter() override {}

private:
  void analyze(edm::Event const &, edm::EventSetup const &) override;
  void beginRun(edm::Run const &, edm::EventSetup const &) override;

  EcalDQMChannelStatus channelStatus_;
  EcalDQMTowerStatus towerStatus_;
  unsigned firstRun_;
  std::ifstream inputFile_;

  EcalElectronicsMapping const *electronicsMap;
  void setElectronicsMap(edm::EventSetup const &);
  EcalElectronicsMapping const *GetElectronicsMap();
};

#endif
