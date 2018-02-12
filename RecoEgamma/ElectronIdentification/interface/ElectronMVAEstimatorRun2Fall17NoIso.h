#ifndef RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Fall17NoIso_H
#define RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Fall17NoIso_H

#include "RecoEgamma//ElectronIdentification/interface/ElectronMVAEstimatorRun2Fall17.h"

class ElectronMVAEstimatorRun2Fall17NoIso : public ElectronMVAEstimatorRun2Fall17 {

 public:

  ElectronMVAEstimatorRun2Fall17NoIso(const edm::ParameterSet& conf) : ElectronMVAEstimatorRun2Fall17(conf, false) {} // False for no isolation
  ~ElectronMVAEstimatorRun2Fall17NoIso() override {}

};

#endif
