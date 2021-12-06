#ifndef DataFormats_GEMDigi_GEMVFATStatusCollection_h
#define DataFormats_GEMDigi_GEMVFATStatusCollection_h

#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMVFATStatus.h"

typedef MuonDigiCollection<GEMDetId, GEMVFATStatus> GEMVFATStatusCollection;

#endif
