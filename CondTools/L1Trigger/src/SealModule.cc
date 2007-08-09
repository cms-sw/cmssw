#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CondTools/L1Trigger/src/L1TDBESSource.h"
#include "CondTools/L1Trigger/src/L1TWriter.h"

DEFINE_FWK_EVENTSETUP_SOURCE(l1t::L1TDBESSource);
DEFINE_FWK_MODULE(l1t::L1TWriter);

