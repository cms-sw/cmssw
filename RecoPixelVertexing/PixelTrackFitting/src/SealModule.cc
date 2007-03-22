#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_SEAL_MODULE();

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackProducer.h"
DEFINE_ANOTHER_FWK_MODULE(PixelTrackProducer);

//#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelHitPairTrackProducer.h"
//DEFINE_ANOTHER_FWK_MODULE(PixelHitPairTrackProducer);


#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterByKinematics.h"
DEFINE_SEAL_PLUGIN(PixelTrackFilterFactory, PixelTrackFilterByKinematics, "PixelTrackFilterByKinematics");

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByConformalMappingAndLine.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByHelixProjections.h"
DEFINE_SEAL_PLUGIN(PixelFitterFactory, PixelFitterByConformalMappingAndLine, "PixelFitterByConformalMappingAndLine");
DEFINE_SEAL_PLUGIN(PixelFitterFactory, PixelFitterByHelixProjections, "PixelFitterByHelixProjections");

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerBySharedHits.h"
DEFINE_SEAL_PLUGIN(PixelTrackCleanerFactory, PixelTrackCleanerBySharedHits, "PixelTrackCleanerBySharedHits");
