#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "PhysicsTools/MVAComputer/interface/MVAComputerESSourceImpl.h"
#include "PhysicsTools/MVATrainer/interface/MVATrainerSaveImpl.h"

#include "CondFormats/DataRecord/interface/PhotonConversionMVAComputerRcd.h"
#include "PhysicsTools/MVAComputer/interface/HelperMacros.h"

using namespace PhysicsTools;

typedef MVAComputerESSourceImpl<PhotonConversionMVAComputerRcd> PhotonConversionMVAComputerESSource;
DEFINE_FWK_EVENTSETUP_SOURCE(PhotonConversionMVAComputerESSource);

typedef MVATrainerContainerSaveImpl<PhotonConversionMVAComputerRcd> PhotonConversionMVAComputerSave;
DEFINE_FWK_MODULE(PhotonConversionMVAComputerSave);

MVA_COMPUTER_CONTAINER_FILE_SOURCE_IMPLEMENT(PhotonConversionMVAComputerRcd, PhotonConversionMVAComputerFileSource);
// eof
