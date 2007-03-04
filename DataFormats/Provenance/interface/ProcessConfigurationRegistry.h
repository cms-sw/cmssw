#ifndef DataFormats_Provenance_ProcessConfgurationRegistry_h
#define DataFormats_Provenance_ProcessConfgurationRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Provenance/interface/ProcessConfguration.h"
#include "DataFormats/Provenance/interface/ProcessConfgurationID.h"

namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::ProcessConfgurationID,edm::ProcessConfguration> ProcessConfgurationRegistry;
  typedef ProcessConfgurationRegistry::collection_type ProcessConfgurationMap;
}

#endif
