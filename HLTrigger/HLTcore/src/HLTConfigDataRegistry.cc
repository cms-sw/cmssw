#include "HLTrigger/HLTcore/interface/HLTConfigProvider.h"
#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.icc"

typedef edm::detail::ThreadSafeRegistry<edm::ParameterSetID, HLTConfigData> HLTConfigDataRegistry;

DEFINE_THREAD_SAFE_REGISTRY_INSTANCE(HLTConfigDataRegistry)
