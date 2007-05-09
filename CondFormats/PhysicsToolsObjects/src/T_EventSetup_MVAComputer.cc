#include "CondFormats/PhysicsToolsObjects/interface/MVAComputer.h"
#include "FWCore/Framework/interface/eventsetupdata_registration_macro.h"

using namespace PhysicsTools::Calibration;

EVENTSETUP_DATA_REG(MVAComputer);

// remove this as soon as you can retrieve individual objects by label
EVENTSETUP_DATA_REG(MVAComputerContainer);
