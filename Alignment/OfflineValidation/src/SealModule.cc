#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "Alignment/OfflineValidation/interface/MuonAlignmentAnalyzer.h"

#include "Alignment/OfflineValidation/interface/TrackerGeometryCompare.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(MuonAlignmentAnalyzer);
DEFINE_ANOTHER_FWK_MODULE(TrackerGeometryCompare);
