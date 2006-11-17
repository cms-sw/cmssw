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

#include "CondFormats/EcalObjects/interface/EcalDCUTemperatures.h"
namespace {
  namespace {
    EcalDCUTemperatures dcuTemperatures;
    std::map<uint32_t, float> dcuTempMap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalPTMTemperatures.h"
namespace {
  namespace {
    EcalPTMTemperatures ptmTemperatures;
    std::map<uint32_t, float> ptmTempMap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"
#include "CondFormats/EcalObjects/interface/EcalChannelStatusCode.h"
namespace {
  namespace {
    EcalChannelStatus channelStatus;
    std::map<uint32_t, EcalChannelStatusCode> statusMap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalMonitoringCorrections.h"
namespace {
  namespace {
    EcalMonitoringCorrections monitorCorrections;
    std::map<uint32_t, EcalMonitoringCorrections::EcalMonitoringCorrection> monCorrectionMap;
  }
}
