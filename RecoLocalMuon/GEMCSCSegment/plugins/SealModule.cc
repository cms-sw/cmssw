#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentProducer.h>
#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegmentBuilderPluginFactory.h>
#include <RecoLocalMuon/GEMCSCSegment/plugins/GEMCSCSegAlgoRR.h>

DEFINE_FWK_MODULE(GEMCSCSegmentProducer);
DEFINE_EDM_PLUGIN(GEMCSCSegmentBuilderPluginFactory, GEMCSCSegAlgoRR, "GEMCSCSegAlgoRR");
