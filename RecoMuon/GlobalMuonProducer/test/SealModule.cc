#include "FWCore/PluginManager/interface/ModuleDef.h"

#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/Services/src/SiteLocalConfigService.h"
#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "RecoMuon/GlobalMuonProducer/test/GLBMuonAnalyzer.h"

using namespace edm::serviceregistry;



DEFINE_FWK_MODULE(GLBMuonAnalyzer);
