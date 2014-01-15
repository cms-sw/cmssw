#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolaFactory.h"
#include "GeneratorInterface/TauolaInterface/interface/TauolappInterface.h"

DEFINE_EDM_PLUGIN(TauolaFactory, gen::TauolappInterface, "Tauolapp105");
