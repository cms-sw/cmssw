#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/BTagObjects/interface/BTagCalibration.h"
#include "CondFormats/DataRecord/interface/BTagCalibrationRcd.h"

using namespace PhysicsTools::Calibration;

REGISTER_PLUGIN(BTagCalibrationRcd, BTagCalibration);
