#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotOnlineObjects.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotHLLHCObjects.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineHLTObjectsRcd.h"
#include "CondFormats/DataRecord/interface/BeamSpotOnlineLegacyObjectsRcd.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotObjectsRcd.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotHLLHCObjectsRcd.h"

REGISTER_PLUGIN(BeamSpotObjectsRcd, BeamSpotObjects);
REGISTER_PLUGIN(BeamSpotOnlineHLTObjectsRcd, BeamSpotOnlineObjects);
REGISTER_PLUGIN_NO_SERIAL(BeamSpotOnlineLegacyObjectsRcd, BeamSpotOnlineObjects);
REGISTER_PLUGIN(SimBeamSpotObjectsRcd, SimBeamSpotObjects);
REGISTER_PLUGIN(SimBeamSpotHLLHCObjectsRcd, SimBeamSpotHLLHCObjects);
