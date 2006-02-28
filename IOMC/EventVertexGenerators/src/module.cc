
#include "IOMC/EventVertexGenerators/interface/BaseEventVertexGenerator.h"
#include "IOMC/EventVertexGenerators/interface/VertexGenerator.h"

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


DEFINE_SEAL_MODULE ();

using edm::VertexGenerator;
DEFINE_ANOTHER_FWK_MODULE(VertexGenerator) ;
