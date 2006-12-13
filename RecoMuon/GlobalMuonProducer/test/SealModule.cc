#include "PluginManager/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "RecoMuon/GlobalMuonProducer/test/Monitor/GlobalMuonModuleMonitor.h"
#include "RecoMuon/GlobalMuonProducer/test/GlobalMuonValidator.h"

using namespace edm::serviceregistry;

DEFINE_SEAL_MODULE();

typedef edm::serviceregistry::AllArgsMaker<GlobalMuonMonitorInterface,GlobalMuonModuleMonitor> MuonModuleMaker;
DEFINE_ANOTHER_FWK_SERVICE_MAKER(GlobalMuonModuleMonitor,MuonModuleMaker);
DEFINE_ANOTHER_FWK_MODULE(GlobalMuonValidator);
