#ifndef DataFormats_Provenance_BranchMapperID_h
#define DataFormats_Provenance_BranchMapperID_h

#include "DataFormats/Provenance/interface/HashedTypes.h"
#include "DataFormats/Provenance/interface/Hash.h"

namespace edm
{
  typedef Hash<BranchMapperType> BranchMapperID;
}


#endif
