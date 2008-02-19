#include <boost/cstdint.hpp>

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

namespace{
  namespace{
    uint32_t i32;
  }
}
#include "CondFormats/EcalObjects/interface/EcalPedestals.h"
namespace {
  EcalPedestals pedmap;
}

#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"
#include "CondFormats/EcalObjects/interface/EcalXtalGroupId.h"
namespace {
  namespace {
    EcalWeightXtalGroups gg;
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
  }
}

#include "CondFormats/EcalObjects/interface/EcalIntercalibConstants.h"
namespace {
  namespace {
    EcalIntercalibConstants intercalib;
  }
}

#include "CondFormats/EcalObjects/interface/EcalIntercalibErrors.h"
namespace {
  namespace {
    EcalIntercalibErrors intercalibErrors;
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
  }
}

#include "CondFormats/EcalObjects/interface/EcalLaserAlphas.h"
namespace {
  namespace {
    EcalLaserAlphas laserAplhas;
  }
}

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"
namespace {
  namespace {
     EcalLaserAPDPNRatios laser_map;
     std::vector<EcalLaserAPDPNRatios::EcalLaserTimeStamp> time_map ;
  }
}

#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatiosRef.h"
namespace {
  namespace {
    EcalLaserAPDPNRatiosRef laserAPDPNRatiosRef;
  }
}

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainEBIdMap.h"
namespace {
  namespace {
    EcalTPGFineGrainConstEB grain;
    std::map<uint32_t, EcalTPGFineGrainConstEB::EcalTPGFineGrainConstEB> EcalTPGFineGrainEBMap ;
  }
}

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainStripEE.h"
namespace {
  namespace {   
    std::map< uint32_t, EcalTPGFineGrainStripEE::Item > EcalTPGFineGrainStripEEMap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalTPGFineGrainTowerEE.h"
namespace {
  namespace {   
    std::map< uint32_t, uint32_t> EcalTPGFineGrainTowerEEMap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalTPGGroups.h"
namespace {
  namespace {   
     std::map<uint32_t, uint32_t> EcalTPGGroupsMap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalTPGLinearizationConst.h"
namespace {
  namespace {
    EcalTPGLinearizationConst tpglinconstmap;
    std::vector<EcalTPGLinearizationConstant> v_tpglinconst;
  }
}

#include "CondFormats/EcalObjects/interface/EcalTPGLutIdMap.h"
namespace {
  namespace {   
    EcalTPGLut lut;
    std::map< uint32_t, EcalTPGLut::EcalTPGLut > EcalTPGLutMap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalTPGPedestals.h"
namespace {
  EcalTPGPedestals tpgpedmap;
  std::vector<EcalTPGPedestal> v_tpgped;
}



#include "CondFormats/EcalObjects/interface/EcalTPGWeightIdMap.h"
namespace {
  namespace {   
    EcalTPGWeights weights;
    std::map<uint32_t, EcalTPGWeights::EcalTPGWeights> EcalTPGWeightMap;
  }
}

#include "CondFormats/EcalObjects/interface/EcalTPGSlidingWindow.h"
namespace {
  namespace {   
   std::map<uint32_t, uint32_t> EcalTPGSlidingWindowMap;
  }
}

