#include <boost/cstdint.hpp>
namespace{
  namespace{
    uint32_t i32;
  }
}
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
namespace {
  std::map< uint32_t, EcalPedestals::Item > pedmap;
}

#include "CondFormats/EcalObjects/interface/EcalWeightRecAlgoWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeight.h"
namespace {
  namespace {
    std::vector< std::vector<EcalWeight> > vecOfVec0;
    std::vector<EcalWeight>  vec0;
  }
}
#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
namespace {
  namespace {
    EcalWeightXtalGroups  gg;
    std::map<uint32_t, EcalXtalGroupId> groupmap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
namespace {
  namespace {
    EcalTBWeights tbwgt;
    EcalWeightSet wset;
    EcalTBWeights::EcalTDCId id;
    std::map< std::pair< EcalXtalGroupId, EcalTBWeights::EcalTDCId > , EcalWeightSet > wgmap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"
namespace {
  namespace {
    EcalADCToGeVConstant adcfactor;
  }
}

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"
#include "CondFormats/EcalObjects/interface/EcalMGPAGainRatio.h"
namespace {
  namespace {
    EcalGainRatios gainratios;
    std::map<uint32_t, EcalMGPAGainRatio> ratiomap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
namespace {
  namespace {
    EcalIntercalibConstants intercalib;
    std::map<uint32_t, EcalIntercalibConstants::EcalIntercalibConstant> intermap;
  }
}
