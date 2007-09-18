#include "CondCore/PluginSystem/interface/registration_macros.h"
#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCalibration.h"
#include "CondFormats/DataRecord/interface/ElectronLikelihoodPdfsRcd.h"

DEFINE_SEAL_MODULE();
REGISTER_PLUGIN(ElectronLikelihoodPdfsRcd,ElectronLikelihoodCalibration);
