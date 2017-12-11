#ifndef ElectronMVAEstimatorRun2Fall17NoIso_h
#define ElectronMVAEstimatorRun2Fall17NoIso_h

#include "RecoEgamma//ElectronIdentification/interface/ElectronMVAEstimatorRun2Fall17.h"

class ElectronMVAEstimatorRun2Fall17NoIso : public ElectronMVAEstimatorRun2Fall17 {

 public:

  ElectronMVAEstimatorRun2Fall17NoIso(const edm::ParameterSet& conf) : ElectronMVAEstimatorRun2Fall17(conf, false) {} // False for no isolation
  ~ElectronMVAEstimatorRun2Fall17NoIso() override {}

};

#endif
