#ifndef DataFormats_GEMDigi_GEMAMCdataCollection_h
#define DataFormats_GEMDigi_GEMAMCdataCollection_h

#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "EventFilter/GEMRawToDigi/interface/AMCdata.h"

typedef MuonDigiCollection<uint16_t, gem::AMCdata> GEMAMCdataCollection;

#endif
