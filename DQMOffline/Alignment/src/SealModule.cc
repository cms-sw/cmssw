#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/PluginManager/interface/ModuleDef.h"

// the clients
#include "DQMOffline/Alignment/interface/MuonAlignment.h"
#include "DQMOffline/Alignment/interface/MuonAlignmentSummary.h"

DEFINE_FWK_MODULE(MuonAlignment);
DEFINE_FWK_MODULE(MuonAlignmentSummary);
