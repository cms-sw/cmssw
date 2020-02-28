#ifndef DataFormats_GEMDigi_GEMGEBdataCollection_h
#define DataFormats_GEMDigi_GEMGEBdataCollection_h

#include "EventFilter/GEMRawToDigi/interface/GEBdata.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<GEMDetId, gem::GEBdata> GEMGEBdataCollection;

#endif
