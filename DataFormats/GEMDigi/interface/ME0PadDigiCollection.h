#ifndef DataFormats_GEMDigi_ME0PadDigiCollection_h
#define DataFormats_GEMDigi_ME0PadDigiCollection_h

/** \class ME0PadDigiCollection
 *  
 *  \author Sven Dildick
 */

#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<ME0DetId, ME0PadDigi> ME0PadDigiCollection;

#endif

