#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "GeneratorInterface/TauolaInterface/interface/TauolaInterface.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaFactory.h"

DEFINE_EDM_PLUGIN(TauolaFactory, gen::TauolaInterface, "Tauola27");
