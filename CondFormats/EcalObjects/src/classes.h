#include <boost/cstdint.hpp>
namespace {
  namespace {
    uint32_t i32;
  }
}


#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
template std::map< uint32_t, EcalPedestals::Item >::iterator;
template std::map< uint32_t, EcalPedestals::Item >::const_iterator;

#include "CondFormats/EcalObjects/interface/EcalWeightRecAlgoWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeight.h"
namespace {
  namespace {
    std::vector< std::vector<EcalWeight> > vecOfVec0;
    std::vector<EcalWeight>  vec0;
  }
}
template  std::vector<EcalWeight>::iterator;
template  std::vector< std::vector<EcalWeight> >::iterator;


#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
namespace {
  namespace {
    EcalWeightXtalGroups  gg;
  }
}
template std::map<uint32_t, EcalXtalGroupId>::iterator;
template std::map<uint32_t, EcalXtalGroupId>::const_iterator;

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"
#include "CondFormats/EcalObjects/interface/EcalWeightSet.h"
namespace {
  namespace {
    EcalTBWeights tbwgt;
    EcalWeightSet wset;
    EcalTDCId id;
  }
}
template std::map< std::pair< EcalXtalGroupId, EcalTDCId >, EcalWeightSet >::iterator;
template std::map< std::pair< EcalXtalGroupId, EcalTDCId >, EcalWeightSet >::const_iterator;

//#include "CondFormats/EcalObjects/interface/.h"
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
  }
}
template std::map<uint32_t, EcalMGPAGainRatio>::iterator;
template std::map<uint32_t, EcalMGPAGainRatio>::const_iterator;


#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
namespace {
  namespace {
    EcalIntercalibConstants intercalib;
  }
}
template std::map<uint32_t, EcalIntercalibConstant>::iterator;
template std::map<uint32_t, EcalIntercalibConstant>::const_iterator;
