
#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "JetTracksAssociatorInVertex.h"
#include "JetTracksAssociatorAtCaloFace.h"
#include "JetExtender.h"


DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(JetTracksAssociatorInVertex);
DEFINE_ANOTHER_FWK_MODULE(JetTracksAssociatorAtCaloFace);
DEFINE_ANOTHER_FWK_MODULE(JetExtender);
