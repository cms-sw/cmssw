#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaFactory.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolappInterface.h"

#ifndef UseTauola114
DEFINE_EDM_PLUGIN(TauolaFactory, gen::TauolappInterface, "Tauolapp105");
#endif

#ifdef UseTauola114
DEFINE_EDM_PLUGIN(TauolaFactory, gen::TauolappInterface, "Tauolapp114");
#endif
