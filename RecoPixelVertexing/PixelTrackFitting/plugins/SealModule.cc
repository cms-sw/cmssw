#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"


#include "PixelTrackProducer.h"
DEFINE_FWK_MODULE(PixelTrackProducer);

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackFilterByKinematics.h"
DEFINE_EDM_PLUGIN(PixelTrackFilterFactory, PixelTrackFilterByKinematics, "PixelTrackFilterByKinematics");

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitter.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByConformalMappingAndLine.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelFitterByHelixProjections.h"
DEFINE_EDM_PLUGIN(PixelFitterFactory, PixelFitterByConformalMappingAndLine, "PixelFitterByConformalMappingAndLine");
DEFINE_EDM_PLUGIN(PixelFitterFactory, PixelFitterByHelixProjections, "PixelFitterByHelixProjections");

#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleaner.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerFactory.h"
#include "RecoPixelVertexing/PixelTrackFitting/interface/PixelTrackCleanerBySharedHits.h"
DEFINE_EDM_PLUGIN(PixelTrackCleanerFactory, PixelTrackCleanerBySharedHits, "PixelTrackCleanerBySharedHits");
