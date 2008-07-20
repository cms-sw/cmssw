#include "CondCore/Utilities/interface/PayLoadInspector.h"
#include "CondCore/Utilities/interface/InspectorPythonWrapper.h"


#include "CondFormats/EcalObjects/interface/EcalWeightXtalGroups.h"

#include "CondFormats/EcalObjects/interface/EcalTBWeights.h"

#include "CondFormats/EcalObjects/interface/EcalGainRatios.h"

#include "CondFormats/EcalObjects/interface/EcalCondObjectContainer.h"

#include "CondFormats/EcalObjects/interface/EcalADCToGeVConstant.h"




#include "CondFormats/EcalObjects/interface/EcalLaserAPDPNRatios.h"


#include "CondFormats/EcalObjects/interface/EcalChannelStatus.h"


#include "CondFormats/EcalObjects/interface/EcalMappingElectronics.h"





namespace { const char * pluginName_ = "pluginEcalDummiesPyInterface"; }

BOOST_PYTHON_MODULE(pluginEcalDummiesPyInterface) {
    define<cond::PayLoadInspector<EcalWeightXtalGroups>();
    define<cond::PayLoadInspector<EcalTBWeights>();
    define<cond::PayLoadInspector<EcalGainRatios>();
    define<cond::PayLoadInspector<EcalFloatCondObjectContainer>();
    define<cond::PayLoadInspector<EcalFloatCondObjectContainer>();
    define<cond::PayLoadInspector<EcalADCToGeVConstant>();
    define<cond::PayLoadInspector<EcalFloatCondObjectContainer>();
    define<cond::PayLoadInspector<EcalLaserAPDPNRatios>();
    define<cond::PayLoadInspector<EcalFloatCondObjectContainer>();
    define<cond::PayLoadInspector<EcalChannelStatus>();
    define<cond::PayLoadInspector<EcalMappingElectronics>();
}

PYTHON_ID(EcalWeightXtalGroups,pluginName_);
PYTHON_ID(EcalTBWeights,pluginName_);
PYTHON_ID(EcalGainRatios,pluginName_);
PYTHON_ID(EcalFloatCondObjectContainer,pluginName_);
PYTHON_ID(EcalFloatCondObjectContainer,pluginName_);
PYTHON_ID(EcalADCToGeVConstant,pluginName_);
PYTHON_ID(EcalFloatCondObjectContainer,pluginName_);
PYTHON_ID(EcalLaserAPDPNRatios,pluginName_);
PYTHON_ID(EcalFloatCondObjectContainer,pluginName_);
PYTHON_ID(EcalChannelStatus,pluginName_);
PYTHON_ID(EcalMappingElectronics,pluginName_);



