#ifndef DataFormats_GEMDigi_GEMOHStatusCollection_h
#define DataFormats_GEMDigi_GEMOHStatusCollection_h

#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMOHStatus.h"

typedef MuonDigiCollection<GEMDetId, GEMOHStatus> GEMOHStatusCollection;

#endif
