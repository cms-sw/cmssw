#include <FWCore/PluginManager/interface/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentProducer.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegmentBuilderPluginFactory.h>

#include <RecoLocalMuon/CSCSegment/src/CSCSegAlgoSK.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegAlgoTC.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegAlgoDF.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegAlgoST.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegAlgoRU.h>


DEFINE_FWK_MODULE(CSCSegmentProducer);
DEFINE_EDM_PLUGIN(CSCSegmentBuilderPluginFactory, CSCSegAlgoSK, "CSCSegAlgoSK");
DEFINE_EDM_PLUGIN(CSCSegmentBuilderPluginFactory, CSCSegAlgoTC, "CSCSegAlgoTC");
DEFINE_EDM_PLUGIN(CSCSegmentBuilderPluginFactory, CSCSegAlgoDF, "CSCSegAlgoDF");
DEFINE_EDM_PLUGIN(CSCSegmentBuilderPluginFactory, CSCSegAlgoST, "CSCSegAlgoST");
DEFINE_EDM_PLUGIN(CSCSegmentBuilderPluginFactory, CSCSegAlgoRU, "CSCSegAlgoRU");

