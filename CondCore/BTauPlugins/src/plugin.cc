#include "CondCore/ESSources/interface/registration_macros.h"

#include "CondFormats/BTauObjects/interface/TrackProbabilityCalibration.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability2DRcd.h"
#include "CondFormats/DataRecord/interface/BTagTrackProbability3DRcd.h"

#include "CondFormats/BTauObjects/interface/CombinedTauTagCalibration.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "CondFormats/DataRecord/interface/CombinedTauTagRcd.h"
#include "CondFormats/DataRecord/interface/BTauGenericMVAJetTagComputerRcd.h"
#include "CondFormats/DataRecord/interface/TauTagMVAComputerRcd.h"

using namespace PhysicsTools::Calibration;



REGISTER_PLUGIN(CombinedTauTagRcd,CombinedTauTagCalibration);
REGISTER_PLUGIN(BTauGenericMVAJetTagComputerRcd, MVAComputerContainer);
REGISTER_PLUGIN(TauTagMVAComputerRcd, MVAComputerContainer);
REGISTER_PLUGIN(BTagTrackProbability2DRcd,TrackProbabilityCalibration);
REGISTER_PLUGIN(BTagTrackProbability3DRcd,TrackProbabilityCalibration);
