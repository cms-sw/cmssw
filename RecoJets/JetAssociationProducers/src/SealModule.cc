
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "JetTracksAssociatorAtVertex.h"
#include "JetTracksAssociatorAtCaloFace.h"
#include "JetExtender.h"


DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(JetTracksAssociatorAtVertex);
DEFINE_ANOTHER_FWK_MODULE(JetTracksAssociatorAtCaloFace);
DEFINE_ANOTHER_FWK_MODULE(JetExtender);
