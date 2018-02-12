#ifndef RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Fall17Iso_H
#define RecoEgamma_ElectronIdentification_ElectronMVAEstimatorRun2Fall17Iso_H

#include "RecoEgamma/ElectronIdentification/interface/ElectronMVAEstimatorRun2Fall17.h"

class ElectronMVAEstimatorRun2Fall17Iso : public ElectronMVAEstimatorRun2Fall17 {

 public:

  ElectronMVAEstimatorRun2Fall17Iso(const edm::ParameterSet& conf) : ElectronMVAEstimatorRun2Fall17(conf, true) {} // True for with isolation
  ~ElectronMVAEstimatorRun2Fall17Iso() override {}

};

#endif
