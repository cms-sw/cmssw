
//#include "IOMC/EventVertexGenerators/interface/BaseEventVertexGenerator.h"
//#include "IOMC/EventVertexGenerators/interface/VertexGenerator.h"

#include "IOMC/EventVertexGenerators/interface/BaseEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/PassThroughEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/GaussEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/FlatEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/BeamProfileVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/BetafuncEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/GaussianZBeamSpotFilter.h"
#include "IOMC/EventVertexGenerators/interface/HLLHCEvtVtxGenerator.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"




//using edm::VertexGenerator;
//DEFINE_FWK_MODULE(VertexGenerator) ;
DEFINE_FWK_MODULE(PassThroughEvtVtxGenerator) ;
DEFINE_FWK_MODULE(GaussEvtVtxGenerator) ;
DEFINE_FWK_MODULE(FlatEvtVtxGenerator) ;
DEFINE_FWK_MODULE(BeamProfileVtxGenerator) ;
DEFINE_FWK_MODULE(BetafuncEvtVtxGenerator) ;
DEFINE_FWK_MODULE(GaussianZBeamSpotFilter);
DEFINE_FWK_MODULE(HLLHCEvtVtxGenerator);
