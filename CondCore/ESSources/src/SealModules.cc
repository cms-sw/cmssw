#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CondCore/ESSources/interface/EcalPedestalsRetriever.h"
#include "CondCore/ESSources/interface/TrackerPedestalsRetriever.h"
#include "CondCore/ESSources/interface/TrackerAlignmentRetriever.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(EcalPedestalsRetriever)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(TrackerPedestalsRetriever)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(TrackerAlignmentRetriever)
