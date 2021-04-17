#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalMuon/GEMSegment/plugins/ME0SegmentBuilderPluginFactory.h"
#include "RecoLocalMuon/GEMSegment/plugins/ME0SegmentAlgorithm.h"
#include "RecoLocalMuon/GEMSegment/plugins/ME0SegAlgoRU.h"
#include "RecoLocalMuon/GEMSegment/plugins/GE0SegAlgoRU.h"
#include "RecoLocalMuon/GEMSegment/plugins/GEMSegmentBuilderPluginFactory.h"
#include "RecoLocalMuon/GEMSegment/plugins/GEMSegmentAlgorithm.h"

DEFINE_EDM_PLUGIN(GEMSegmentBuilderPluginFactory, GEMSegmentAlgorithm, "GEMSegmentAlgorithm");
DEFINE_EDM_PLUGIN(GEMSegmentBuilderPluginFactory, GE0SegAlgoRU, "GE0SegAlgoRU");
DEFINE_EDM_PLUGIN(ME0SegmentBuilderPluginFactory, ME0SegmentAlgorithm, "ME0SegmentAlgorithm");
DEFINE_EDM_PLUGIN(ME0SegmentBuilderPluginFactory, ME0SegAlgoRU, "ME0SegAlgoRU");
