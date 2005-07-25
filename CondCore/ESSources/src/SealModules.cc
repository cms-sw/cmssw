#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CondCore/ESSources/interface/PedestalRetriever.h"
#include "CondCore/ESSources/interface/AlignmentRetriever.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(PedestalRetriever)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(AlignmentRetriever)
