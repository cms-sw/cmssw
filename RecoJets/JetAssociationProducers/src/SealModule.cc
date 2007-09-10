
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "JetToTracksAssociator.h"
#include "JetToTracksAssociatorAtCaloFace.h"


DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(JetToTracksAssociator);
DEFINE_ANOTHER_FWK_MODULE(JetToTracksAssociatorAtCaloFace);
