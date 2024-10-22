#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"
#include "DQMServices/Core/interface/DQMStore.h"
#include "DQMServices/Core/src/DQMService.h"

DEFINE_FWK_SERVICE_MAKER(DQM, edm::serviceregistry::AllArgsMaker<DQMService>);
DEFINE_FWK_SERVICE_MAKER(DQMStore, edm::serviceregistry::AllArgsMaker<dqm::legacy::DQMStore>);
