#include "RecoEgamma//ElectronIdentification/interface/ElectronMVAEstimatorRun2Fall17.h"

class ElectronMVAEstimatorRun2Fall17noIso : public ElectronMVAEstimatorRun2Fall17 {

 public:

  ElectronMVAEstimatorRun2Fall17noIso(const edm::ParameterSet& conf) : ElectronMVAEstimatorRun2Fall17(conf, false) {} // False for no isolation
  ~ElectronMVAEstimatorRun2Fall17noIso() override {}

  const std::string& getName() const final { return name_; }

 private:

  const std::string name_ = "ElectronMVAEstimatorRun2Fall17noIso";

};
