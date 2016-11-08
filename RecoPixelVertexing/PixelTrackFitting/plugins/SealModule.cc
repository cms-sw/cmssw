#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "PixelTrackProducer.h"
DEFINE_FWK_MODULE(PixelTrackProducer);

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerBySharedHits.h"
DEFINE_EDM_PLUGIN(PixelTrackCleanerFactory, PixelTrackCleanerBySharedHits, "PixelTrackCleanerBySharedHits");
