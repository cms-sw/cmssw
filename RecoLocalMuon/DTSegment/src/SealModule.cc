#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoLocalMuon/DTSegment/src/DTRecSegment2DProducer.h"
#include "RecoLocalMuon/DTSegment/src/DTRecSegment2DAlgoFactory.h"
#include "RecoLocalMuon/DTSegment/src/DTCombinatorialPatternReco.h"


DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(DTRecSegment2DProducer);
DEFINE_SEAL_PLUGIN (DTRecSegment2DAlgoFactory, DTCombinatorialPatternReco, "DTCombinatorialPatternReco");
