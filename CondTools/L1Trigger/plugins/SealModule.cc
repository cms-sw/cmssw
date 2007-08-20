#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CondTools/L1Trigger/plugins/L1TDBESSource.h"
#include "CondTools/L1Trigger/plugins/L1TWriter.h"

using namespace l1t;

DEFINE_FWK_EVENTSETUP_SOURCE(L1TDBESSource);
DEFINE_FWK_MODULE(L1TWriter);

