#ifndef DataFormats_GEMDigi_GEMVfatStatusDigiCollection_h
#define DataFormats_GEMDigi_GEMVfatStatusDigiCollection_h

#include "EventFilter/GEMRawToDigi/interface/GEMVfatStatusDigi.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<GEMDetId, GEMVfatStatusDigi> GEMVfatStatusDigiCollection;

#endif
