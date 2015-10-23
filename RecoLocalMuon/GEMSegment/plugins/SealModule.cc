#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <RecoLocalMuon/GEMSegment/plugins/ME0SegmentProducer.h>
#include <RecoLocalMuon/GEMSegment/plugins/ME0SegmentBuilderPluginFactory.h>
#include <RecoLocalMuon/GEMSegment/plugins/ME0SegAlgoMM.h>

DEFINE_FWK_MODULE(ME0SegmentProducer);
DEFINE_EDM_PLUGIN(ME0SegmentBuilderPluginFactory, ME0SegAlgoMM, "ME0SegAlgoMM");
