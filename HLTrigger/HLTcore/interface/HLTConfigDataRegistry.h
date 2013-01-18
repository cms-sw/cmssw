#ifndef HLTConfigDataRegistry_H
#define HLTConfigDataRegistry_H

#include "DataFormats/Provenance/interface/ParameterSetID.h"
#include "HLTrigger/HLTcore/interface/HLTConfigData.h"
#include "HLTrigger/HLTcore/interface/HLTConfigCounter.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "FWCore/Utilities/interface/ThreadSafeRegistry.icc"

typedef edm::detail::ThreadSafeRegistry<edm::ParameterSetID, HLTConfigData, HLTConfigCounter> HLTConfigDataRegistry;

#endif
