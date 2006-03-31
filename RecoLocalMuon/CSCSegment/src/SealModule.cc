#include <PluginManager/ModuleDef.h>
#include <FWCore/Framework/interface/MakerMacros.h>

#include <RecoLocalMuon/CSCSegment/src/CSCSegmentProducer.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegmentBuilderPluginFactory.h>
#include <RecoLocalMuon/CSCSegment/src/CSCSegAlgoSK.h>

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(CSCSegmentProducer)
DEFINE_SEAL_PLUGIN(CSCSegmentBuilderPluginFactory, CSCSegAlgoSK, "CSCSegAlgoSK");

