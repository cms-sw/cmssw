#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputerRecord.h"
#include "PhysicsTools/MVAComputer/interface/MVAComputerESSourceImpl.h"

using namespace PhysicsTools;

// define ESSource using the dummy MVAComputer record for testing purposes

typedef MVAComputerESSourceImpl<MVAComputerRecord> MVAComputerESSource;

DEFINE_FWK_EVENTSETUP_SOURCE(MVAComputerESSource);
