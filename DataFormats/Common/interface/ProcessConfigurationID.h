#ifndef DataFormatsCommonProcessConfigurationID_h
#define DataFormatsCommonProcessConfigurationID_h

#include "DataFormats/Common/interface/HashedTypes.h"
#include "DataFormats/Common/interface/Hash.h"

namespace edm
{
  typedef Hash<ProcessConfigurationType> ProcessConfigurationID;
}


#endif
