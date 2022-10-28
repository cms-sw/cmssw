#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "DQM/GEM/plugins/GEMEfficiencyAnalyzer.h"
#include "DQM/GEM/plugins/GEMEfficiencyHarvester.h"
#include "DQM/GEM/plugins/GEMEffByGEMCSCSegmentSource.h"
#include "DQM/GEM/plugins/GEMEffByGEMCSCSegmentClient.h"

DEFINE_FWK_MODULE(GEMEfficiencyAnalyzer);
DEFINE_FWK_MODULE(GEMEfficiencyHarvester);
DEFINE_FWK_MODULE(GEMEffByGEMCSCSegmentSource);
DEFINE_FWK_MODULE(GEMEffByGEMCSCSegmentClient);
