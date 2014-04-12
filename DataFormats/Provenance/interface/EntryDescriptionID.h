#ifndef DataFormats_Provenance_EntryDescriptionID_h
#define DataFormats_Provenance_EntryDescriptionID_h

#include "DataFormats/Provenance/interface/HashedTypes.h"
#include "DataFormats/Provenance/interface/Hash.h"

namespace edm
{
  typedef Hash<EntryDescriptionType> EntryDescriptionID;
}


#endif
