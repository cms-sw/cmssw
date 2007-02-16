#include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondFormats/BTagObjects/interface/TrackProbabilityCalibration.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(BTagTrackProbability2DRcd,TrackProbabilityCalibration);
REGISTER_PLUGIN(BTagTrackProbability3DRcd,TrackProbabilityCalibration);
