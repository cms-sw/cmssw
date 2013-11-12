#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "GeneratorInterface/EvtGenInterface/interface/EvtGenFactory.h"
#include "GeneratorInterface/EvtGenInterface/interface/EvtGenInterface.h"

DEFINE_EDM_PLUGIN(EvtGenFactory, gen::EvtGenInterface, "EvtGenLHC91");
