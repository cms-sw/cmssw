#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoPixelVertexing/PixelVertexFinding/interface/PixelVertexProducer.h"
#include "RecoPixelVertexing/PixelVertexFinding/interface/SkipBadEvents.h"


DEFINE_FWK_MODULE(PixelVertexProducer);
DEFINE_FWK_MODULE(SkipBadEvents);
