#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "RecoMuon/GlobalTrackingTools/plugins/GlobalTrackQualityProducer.h"

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE(GlobalTrackQualityProducer);
