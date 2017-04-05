#ifndef ME0LCTDigi_ME0LCTDigiCollection_h
#define ME0LCTDigi_ME0LCTDigiCollection_h

/** \class ME0LCTDigiCollection
 *
 *  \author Sven Dildick (TAMU)
 *
 */

#include <DataFormats/MuonDetId/interface/ME0DetId.h>
#include <DataFormats/GEMDigi/interface/ME0LCTDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<ME0DetId,ME0LCTDigi> ME0LCTDigiCollection;

#endif
