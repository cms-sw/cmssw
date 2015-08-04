
#include "GeneratorInterface/Core/interface/BaseEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/BeamProfileVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/BetaBoostEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/BetafuncEvtVtxGenerator.h"
#include "GeneratorInterface/Core/interface/EventVertexGeneratorFactory.h"
#include "IOMC/EventVertexGenerators/interface/EventVertexProducer.h"
#include "IOMC/EventVertexGenerators/interface/FlatEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/GaussEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/GaussianZBeamSpotFilter.h"
#include "IOMC/EventVertexGenerators/interface/MixBoostEvtVtxGenerator.h"
#include "IOMC/EventVertexGenerators/interface/MixEvtVtxGenerator.h"

#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

DEFINE_EVENTVERTEX_GENERATOR(GaussEvtVtxGenerator);
DEFINE_EVENTVERTEX_GENERATOR(FlatEvtVtxGenerator);
DEFINE_EVENTVERTEX_GENERATOR(BeamProfileVtxGenerator);
DEFINE_EVENTVERTEX_GENERATOR(BetafuncEvtVtxGenerator);
DEFINE_EVENTVERTEX_GENERATOR(BetaBoostEvtVtxGenerator);
DEFINE_EVENTVERTEX_GENERATOR(MixBoostEvtVtxGenerator);
DEFINE_EVENTVERTEX_GENERATOR(MixEvtVtxGenerator);
DEFINE_FWK_MODULE(EventVertexProducer);
DEFINE_FWK_MODULE(GaussianZBeamSpotFilter);
