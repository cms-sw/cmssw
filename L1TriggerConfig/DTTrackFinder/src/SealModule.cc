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
DEFINE_FWK_MODULE(DTPhiLutTester);
DEFINE_FWK_MODULE(DTPtaLutTester);
DEFINE_FWK_MODULE(DTEtaPatternLutTester);
DEFINE_FWK_MODULE(DTQualPatternLutTester);
DEFINE_FWK_MODULE(DTTFParametersTester);
DEFINE_FWK_MODULE(DTTFMasksTester);
