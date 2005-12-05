// ShR 17 Sept 2005: add this until scram can handle packages with only templated classes

#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecAnalFitAlgo.h"
#include "RecoLocalCalo/EcalRecAlgos/interface/EcalUncalibRecHitRecWeightsAlgo.h"
#include "DataFormats/EcalDigi/interface/EBDataFrame.h"

namespace {
  namespace {
    class dummyfile {
     public:
      dummyfile() { };
      virtual ~dummyfile() { };
    };
     EcalUncalibRecHitRecAnalFitAlgo<EBDataFrame> algo1;
     EcalUncalibRecHitRecWeightsAlgo<EBDataFrame> algo2;
  }
}
