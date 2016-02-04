#ifndef DataFormats_Provenance_ProcessHistoryID_h
#define DataFormats_Provenance_ProcessHistoryID_h

#include "DataFormats/Provenance/interface/HashedTypes.h"
#include "DataFormats/Provenance/interface/Hash.h"

namespace edm
{
  typedef Hash<ProcessHistoryType> ProcessHistoryID;
}


#endif
