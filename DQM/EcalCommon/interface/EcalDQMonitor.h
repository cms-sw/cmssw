#ifndef EcalDQMonitor_H
#define EcalDQMonitor_H

#include <string>
#include <vector>

#include "FWCore/Framework/interface/EDAnalyzer.h"

namespace edm{
  class ParameterSet;
}
namespace ecaldqm{
  class DQWorker;
}

class EcalDQMonitor : public edm::EDAnalyzer {
 public:
  EcalDQMonitor(const edm::ParameterSet &);
  virtual ~EcalDQMonitor();

 protected:
  std::string moduleName_;
  const bool mergeRuns_;
  const int verbosity_;
  bool initialized_;
};

#endif
