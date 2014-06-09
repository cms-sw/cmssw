#ifndef GEMDigi_GEMCSCCoPadDigiCollection_h
#define GEMDigi_GEMCSCCoPadDigiCollection_h
/** \class GEMCSCCoPadDigiCollection
 * 
 *  \author Sven Dildick
 */

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/GEMDigi/interface/GEMCSCCoPadDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

// GEMDetId is detId of pad in 1st layer
typedef MuonDigiCollection<GEMDetId, GEMCSCCoPadDigi> GEMCSCCoPadDigiCollection;

#endif

