#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "L1Trigger/CSCTrackFinder/plugins/CSCTFanalyzer.h"
#include "L1Trigger/CSCTrackFinder/plugins/CSCTFTrackProducer.h"
#include "L1Trigger/CSCTrackFinder/plugins/CSCTFCandidateProducer.h"

DEFINE_FWK_MODULE(CSCTFTrackProducer);
DEFINE_FWK_MODULE(CSCTFCandidateProducer);
DEFINE_FWK_MODULE(CSCTFanalyzer);

