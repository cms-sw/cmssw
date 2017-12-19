#ifndef DataFormats_GEMDigi_GEMGEBStatusDigiCollection_h
#define DataFormats_GEMDigi_GEMGEBStatusDigiCollection_h

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMGEBStatusDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<GEMDetId, GEMGEBStatusDigi> GEMGEBStatusDigiCollection;

#endif

