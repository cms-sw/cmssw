#ifndef PFDQMEventSelector_H
#define PFDQMEventSelector_H

#include "FWCore/Framework/interface/one/EDFilter.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "DQMServices/Core/interface/DQMStore.h"

class PFDQMEventSelector : public edm::one::EDFilter<edm::one::SharedResources> {
public:
  typedef dqm::legacy::DQMStore DQMStore;
  typedef dqm::legacy::MonitorElement MonitorElement;

  PFDQMEventSelector(const edm::ParameterSet &);
  ~PFDQMEventSelector() override;

private:
  void beginJob() override;
  bool filter(edm::Event &, edm::EventSetup const &) override;
  void endJob() override;

  bool openInputFile();

  uint64_t nEvents_, nSelectedEvents_;
  bool verbose_;

  std::vector<std::string> folderNames_;
  std::string inputFileName_;
  bool fileOpened_;

  DQMStore *dqmStore_;
};

#endif
