#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2Fall17.h"

class ElectronMVAEstimatorRun2Fall17iso : public ElectronMVAEstimatorRun2Fall17 {

 public:

  ElectronMVAEstimatorRun2Fall17iso(const edm::ParameterSet& conf) : ElectronMVAEstimatorRun2Fall17(conf, true) {} // True for with isolation
  ~ElectronMVAEstimatorRun2Fall17iso() override {}

  const std::string& getName() const final { return name_; }

 private:

  const std::string name_ = "ElectronMVAEstimatorRun2Fall17iso";

};
