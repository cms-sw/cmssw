#ifndef DataFormats_Provenance_ProcessConfigurationRegistry_h
#define DataFormats_Provenance_ProcessConfigurationRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Provenance/interface/ProcessConfiguration.h"
#include "DataFormats/Provenance/interface/ProcessConfigurationID.h"

namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::ProcessConfigurationID,edm::ProcessConfiguration> ProcessConfigurationRegistry;
  typedef ProcessConfigurationRegistry::collection_type ProcessConfigurationMap;
  typedef ProcessConfigurationRegistry::vector_type ProcessConfigurationVector;
}

#endif
