#ifndef DataFormats_GEMDigi_GEMVfatStatusDigiCollection_h
#define DataFormats_GEMDigi_GEMVfatStatusDigiCollection_h

#include "DataFormats/GEMDigi/interface/GEMVfatStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<uint16_t, GEMVfatStatusDigi> GEMVfatStatusDigiCollection;

#endif
