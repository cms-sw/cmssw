#ifndef DataFormats_Provenance_ProcessConfigurationID_h
#define DataFormats_Provenance_ProcessConfigurationID_h

#include "DataFormats/Provenance/interface/HashedTypes.h"
#include "DataFormats/Provenance/interface/Hash.h"

namespace edm
{
  typedef Hash<ProcessConfigurationType> ProcessConfigurationID;
}


#endif
