#ifndef DataFormats_Provenance_ParentageID_h
#define DataFormats_Provenance_ParentageID_h

#include "DataFormats/Provenance/interface/HashedTypes.h"
#include "DataFormats/Provenance/interface/Hash.h"

namespace edm
{
  typedef Hash<ParentageType> ParentageID;
}


#endif
