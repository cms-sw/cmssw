#ifndef GEMDigi_GEMCoPadDigiCollection_h
#define GEMDigi_GEMCoPadDigiCollection_h
/** \class GEMCoPadDigiCollection
 * 
 *  \author Sven Dildick
 */

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMCoPadDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

// GEMDetId is detId of pad in 1st layer
typedef MuonDigiCollection<GEMDetId, GEMCoPadDigi> GEMCoPadDigiCollection;

#endif
