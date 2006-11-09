
#include "IOMC/EventVertexGenerators/interface/BaseEventVertexGenerator.h"
#include "IOMC/EventVertexGenerators/interface/VertexGenerator.h"

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/GaussEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/FlatEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/BeamProfileVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/BetafuncEvtVtxGenerator.h"

#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


DEFINE_SEAL_MODULE ();

using edm::VertexGenerator;
DEFINE_ANOTHER_FWK_MODULE(VertexGenerator) ;
DEFINE_ANOTHER_FWK_MODULE(GaussEvtVtxGenerator) ;
DEFINE_ANOTHER_FWK_MODULE(FlatEvtVtxGenerator) ;
DEFINE_ANOTHER_FWK_MODULE(BeamProfileVtxGenerator) ;
DEFINE_ANOTHER_FWK_MODULE(BetafuncEvtVtxGenerator) ;
