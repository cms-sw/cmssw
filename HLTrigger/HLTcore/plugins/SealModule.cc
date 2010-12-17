#include "FWCore/ServiceRegistry/interface/ServiceMaker.h"

#include "HLTrigger/HLTcore/interface/HLTConfigService.h"
using edm::service::HLTConfigService;
DEFINE_FWK_SERVICE(HLTConfigService);

#include "FWCore/Framework/interface/MakerMacros.h"

#include "HLTrigger/HLTcore/interface/HLTPrescaler.h"
#include "HLTrigger/HLTcore/interface/HLTEventAnalyzerAOD.h"
#include "HLTrigger/HLTcore/interface/HLTEventAnalyzerRAW.h"
#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerAOD.h"
#include "HLTrigger/HLTcore/interface/TriggerSummaryAnalyzerRAW.h"
#include "HLTrigger/HLTcore/interface/TriggerSummaryProducerAOD.h"
#include "HLTrigger/HLTcore/interface/TriggerSummaryProducerRAW.h"
#include "HLTrigger/HLTcore/interface/HLTPrescaleRecorder.h"

DEFINE_FWK_MODULE(HLTPrescaler);
DEFINE_FWK_MODULE(HLTEventAnalyzerAOD);
DEFINE_FWK_MODULE(HLTEventAnalyzerRAW);
DEFINE_FWK_MODULE(TriggerSummaryAnalyzerAOD);
DEFINE_FWK_MODULE(TriggerSummaryAnalyzerRAW);
DEFINE_FWK_MODULE(TriggerSummaryProducerAOD);
DEFINE_FWK_MODULE(TriggerSummaryProducerRAW);
DEFINE_FWK_MODULE(HLTPrescaleRecorder);
