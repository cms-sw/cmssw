#ifndef RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2OTCAModuleParamsHost_h
#define RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2OTCAModuleParamsHost_h

#include "DataFormats/GeometrySurface/interface/SOARotation.h"
#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPEPosition.h"

// The OT modules of the CA, one row per CA module: the Phase2StripCPE parameters and the module surface
GENERATE_SOA_LAYOUT(Phase2OTCAModuleParamsLayout,
                    SOA_COLUMN(phase2StripCPE::ModuleParams, cpe),
                    SOA_COLUMN(float, xerrLocal),  // local position errors xx and yy (xy is zero)
                    SOA_COLUMN(float, yerrLocal),
                    SOA_COLUMN(SOAFrame<float>, frame))

using Phase2OTCAModuleParamsSoA = Phase2OTCAModuleParamsLayout<>;
using Phase2OTCAModuleParamsHost = PortableHostCollection<Phase2OTCAModuleParamsSoA>;

#endif  // RecoLocalTracker_Phase2TrackerRecHits_interface_Phase2OTCAModuleParamsHost_h
