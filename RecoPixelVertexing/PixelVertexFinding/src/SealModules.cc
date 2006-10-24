#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoPixelVertexing/PixelVertexFinding/interface/PixelVertexProducer.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/SkipBadEvents.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(PixelVertexProducer);
DEFINE_ANOTHER_FWK_MODULE(SkipBadEvents);
