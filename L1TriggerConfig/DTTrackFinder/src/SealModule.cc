#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "L1TriggerConfig/DTTrackFinder/interface/DTExtLutTester.h"
#include "L1TriggerConfig/DTTrackFinder/interface/DTPhiLutTester.h"
#include "L1TriggerConfig/DTTrackFinder/interface/DTPtaLutTester.h"
#include "L1TriggerConfig/DTTrackFinder/interface/DTEtaPatternLutTester.h"
#include "L1TriggerConfig/DTTrackFinder/interface/DTQualPatternLutTester.h"
#include "L1TriggerConfig/DTTrackFinder/interface/DTTFParametersTester.h"
#include "L1TriggerConfig/DTTrackFinder/interface/DTTFMasksTester.h"

DEFINE_FWK_MODULE(DTExtLutTester);
DEFINE_ANOTHER_FWK_MODULE(DTPhiLutTester);
DEFINE_ANOTHER_FWK_MODULE(DTPtaLutTester);
DEFINE_ANOTHER_FWK_MODULE(DTEtaPatternLutTester);
DEFINE_ANOTHER_FWK_MODULE(DTQualPatternLutTester);
DEFINE_ANOTHER_FWK_MODULE(DTTFParametersTester);
DEFINE_ANOTHER_FWK_MODULE(DTTFMasksTester);
