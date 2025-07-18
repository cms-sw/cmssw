#ifndef DataFormats_Provenance_ParentageID_h
#define DataFormats_Provenance_ParentageID_h

#include "DataFormats/Provenance/interface/HashedTypes.h"
#include "DataFormats/Provenance/interface/CompactHash.h"
#include "DataFormats/Provenance/interface/Hash.h"

namespace edm {
  using ParentageID = CompactHash<ParentageType>;

  using StoredParentageID = Hash<ParentageType>;
}  // namespace edm

#endif
