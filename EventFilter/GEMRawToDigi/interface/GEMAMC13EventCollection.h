#ifndef DataFormats_GEMDigi_GEMAMC13EventCollection_h
#define DataFormats_GEMDigi_GEMAMC13EventCollection_h

#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "EventFilter/GEMRawToDigi/interface/AMC13Event.h"

typedef MuonDigiCollection<uint16_t, gem::AMC13Event> GEMAMC13EventCollection;

#endif
