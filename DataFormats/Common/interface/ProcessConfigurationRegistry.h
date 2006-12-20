#ifndef DataFormats_Common_ProcessConfgurationRegistry_h
#define DataFormats_Common_ProcessConfgurationRegistry_h

#include "FWCore/Utilities/interface/ThreadSafeRegistry.h"
#include "DataFormats/Common/interface/ProcessConfguration.h"
#include "DataFormats/Common/interface/ProcessConfgurationID.h"

namespace edm
{
  typedef edm::detail::ThreadSafeRegistry<edm::ProcessConfgurationID,edm::ProcessConfguration> ProcessConfgurationRegistry;
  typedef ProcessConfgurationRegistry::collection_type ProcessConfgurationMap;
}

#endif
