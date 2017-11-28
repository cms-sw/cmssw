#include "RecoEgamma//ElectronIdentification/interface/ElectronMVAEstimatorRun2.h"

class ElectronMVAEstimatorRun2Fall17noIso : public ElectronMVAEstimatorRun2 {

 public:

  ElectronMVAEstimatorRun2Fall17noIso(const edm::ParameterSet& conf) : ElectronMVAEstimatorRun2(conf) {}
  ~ElectronMVAEstimatorRun2Fall17noIso() {}

  const std::string& getName() const final { return name_; }

 private:

  const std::string name_ = "ElectronMVAEstimatorRun2Fall17noIso";

};
