#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Base.h"
#include "RecoEgamma/EgammaTools/interface/AnyMVAEstimatorRun2Factory.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

AnyMVAEstimatorRun2Base::AnyMVAEstimatorRun2Base(const edm::ParameterSet& conf)
    : tag_(conf.getParameter<std::string>("mvaTag")),
      nCategories_(conf.getParameter<int>("nCategories")),
      debug_(conf.getUntrackedParameter<bool>("debug", false)) {}

EDM_REGISTER_PLUGINFACTORY(AnyMVAEstimatorRun2Factory, "AnyMVAEstimatorRun2Factory");
