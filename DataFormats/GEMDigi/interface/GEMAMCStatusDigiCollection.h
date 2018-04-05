#ifndef DataFormats_GEMDigi_GEMAMCStatusDigiCollection_h
#define DataFormats_GEMDigi_GEMAMCStatusDigiCollection_h

#include "DataFormats/GEMDigi/interface/GEMAMCStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<uint16_t, GEMAMCStatusDigi> GEMAMCStatusDigiCollection;

#endif

