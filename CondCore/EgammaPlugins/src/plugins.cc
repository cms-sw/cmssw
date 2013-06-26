#include "CondCore/ESSources/interface/registration_macros.h"
#include "CondFormats/EgammaObjects/interface/ElectronLikelihoodCalibration.h"
#include "CondFormats/EgammaObjects/interface/GBRForest.h"
#include "CondFormats/DataRecord/interface/ElectronLikelihoodPdfsRcd.h"
#include "CondFormats/DataRecord/interface/GBRWrapperRcd.h"
#include "CondFormats/DataRecord/interface/PhotonConversionMVAComputerRcd.h"
#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"

using namespace PhysicsTools::Calibration;


REGISTER_PLUGIN(ElectronLikelihoodPdfsRcd,ElectronLikelihoodCalibration);
REGISTER_PLUGIN(GBRWrapperRcd,GBRForest);
REGISTER_PLUGIN(PhotonConversionMVAComputerRcd,MVAComputerContainer);
