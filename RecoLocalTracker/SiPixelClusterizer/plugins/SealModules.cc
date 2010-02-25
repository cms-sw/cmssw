//
// plugins/SealModules.cc
// Using new EDM PluginManager (V.Chiochia, April 2007)
//
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "RecoLocalTracker/SiPixelClusterizer/interface/SiPixelClusterProducer.h"
//
using cms::SiPixelClusterProducer;

DEFINE_FWK_MODULE(SiPixelClusterProducer);

