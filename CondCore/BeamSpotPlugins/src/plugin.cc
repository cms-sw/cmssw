#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/DataRecord/interface/SimBeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
#include "CondFormats/BeamSpotObjects/interface/SimBeamSpotObjects.h"

REGISTER_PLUGIN(BeamSpotObjectsRcd,BeamSpotObjects);
REGISTER_PLUGIN(SimBeamSpotObjectsRcd,SimBeamSpotObjects);
