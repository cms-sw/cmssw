#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "GeneratorInterface/PhotosInterface/interface/PhotosFactory.h"

DEFINE_EDM_PLUGIN(PhotosFactory, gen::PhotosInterface, "Photos2155");
