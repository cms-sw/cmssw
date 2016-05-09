#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <RecoLocalMuon/GEMSegment/plugins/ME0SegmentProducer.h>
#include <RecoLocalMuon/GEMSegment/plugins/ME0SegmentBuilderPluginFactory.h>
#include <RecoLocalMuon/GEMSegment/plugins/ME0SegAlgo.h>

DEFINE_FWK_MODULE(ME0SegmentProducer);

#include <RecoLocalMuon/GEMSegment/plugins/GEMSegmentBuilderPluginFactory.h>
#include <RecoLocalMuon/GEMSegment/plugins/GEMSegmentAlgorithm.h>

DEFINE_EDM_PLUGIN(GEMSegmentBuilderPluginFactory, GEMSegmentAlgorithm, "GEMSegmentAlgorithm");
DEFINE_EDM_PLUGIN(ME0SegmentBuilderPluginFactory, ME0SegAlgo, "ME0SegAlgo");
