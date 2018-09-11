#include "FWCore/Utilities/interface/typelookup.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripClusterParameterEstimator.h"

#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelFakeCPE.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/StripFakeCPE.h"


//--- Now use the Framework macros to set it all up:
//
TYPELOOKUP_DATA_REG(PixelClusterParameterEstimator);
TYPELOOKUP_DATA_REG(StripClusterParameterEstimator);

// TYPELOOKUP_DATA_REG(PixelFakeCPE);
// TYPELOOKUP_DATA_REG(StripFakeCPE);

