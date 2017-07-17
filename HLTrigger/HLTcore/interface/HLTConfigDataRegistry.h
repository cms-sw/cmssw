#ifndef HLTrigger_HLTcore_HLTConfigDataRegistry_H
#define HLTrigger_HLTcore_HLTConfigDataRegistry_H

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.icc"

typedef edm::detail::ThreadSafeRegistry<edm::ParameterSetID, HLTConfigData> HLTConfigDataRegistry;

#endif
