#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/DataRecord/interface/BeamSpotObjectsRcd.h"
#include "CondFormats/BeamSpotObjects/interface/BeamSpotObjects.h"
DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(BeamSpotObjectsRcd,BeamSpotObjects);
