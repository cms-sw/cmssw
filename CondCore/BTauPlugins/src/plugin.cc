#include "CondCore/PluginSystem/interface/registration_macros.h"

#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"

#include "CondFormats/BTauObjects/interface/CombinedTauTagCalibration.h"
#include "CondFormats/DataRecord/interface/CombinedTauTagRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(BTagTrackProbability2DRcd,TrackProbabilityCalibration);
REGISTER_PLUGIN(BTagTrackProbability3DRcd,TrackProbabilityCalibration);
REGISTER_PLUGIN(CombinedTauTagRcd,CombinedTauTagCalibration);
