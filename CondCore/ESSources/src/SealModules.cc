#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/SourceFactory.h"
#include "CondCore/ESSources/interface/EcalPedestalsRetriever.h"
#include "CondCore/ESSources/interface/HcalPedestalsRetriever.h"
#include "CondCore/ESSources/interface/TrackerPedestalsRetriever.h"
#include "CondCore/ESSources/interface/TrackerAlignmentRetriever.h"
#include "CondCore/ESSources/interface/SiStripReadOutCablingRetriever.h"
#include "CondCore/ESSources/interface/DTT0Retriever.h"
#include "CondCore/ESSources/interface/DTReadOutMappingRetriever.h"
DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(EcalPedestalsRetriever)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(HcalPedestalsRetriever)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(TrackerPedestalsRetriever)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(TrackerAlignmentRetriever)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(SiStripReadOutCablingRetriever)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(DTT0Retriever)
DEFINE_ANOTHER_FWK_EVENTSETUP_SOURCE(DTReadOutMappingRetriever)
