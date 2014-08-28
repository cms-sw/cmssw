// Here are the necessary incantations to declare your module to the
// framework, so it can be referenced in a cmsRun file.
//
#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/SourceFactory.h"

#include "HLTrigger/JSONMonitoring/interface/TriggerJSONMonitoring.h"

DEFINE_FWK_MODULE(TriggerJSONMonitoring);
