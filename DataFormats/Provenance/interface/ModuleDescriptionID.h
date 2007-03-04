#ifndef DataFormats_Provenance_ModuleDescriptionID_h
#define DataFormats_Provenance_ModuleDescriptionID_h

#include "DataFormats/Provenance/interface/HashedTypes.h"
#include "DataFormats/Provenance/interface/Hash.h"

namespace edm
{
  typedef Hash<ModuleDescriptionType> ModuleDescriptionID;
}


#endif
