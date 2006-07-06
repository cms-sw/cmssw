#ifndef DataFormatsCommonProcessHistoryID_h
#define DataFormatsCommonProcessHistoryID_h

#include "DataFormats/Common/interface/HashedTypes.h"
#include "DataFormats/Common/interface/Hash.h"

namespace edm
{
  typedef Hash<ProcessHistoryType> ProcessHistoryID;
}


#endif
