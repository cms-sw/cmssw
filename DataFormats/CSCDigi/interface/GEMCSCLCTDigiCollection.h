#ifndef GEMCSCLCTDigi_GEMCSCLCTDigiCollection_h
#define GEMCSCLCTDigi_GEMCSCLCTDigiCollection_h

#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCDigi/interface/GEMCSCLCTDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<CSCDetId,GEMCSCLCTDigi> GEMCSCLCTDigiCollection;

#endif
