#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaFactory.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaInterface.h"

DEFINE_EDM_PLUGIN(TauolaFactory, gen::TauolaInterface, "Tauola-27.121.5");

