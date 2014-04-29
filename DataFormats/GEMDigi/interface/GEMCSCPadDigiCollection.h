#ifndef GEMDigi_GEMCSCPadDigiCollection_h
#define GEMDigi_GEMCSCPadDigiCollection_h
/** \class GEMCSCPadDigiCollection
 *  
 *  \author Vadim Khotilovich
 */

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/GEMDigi/interface/GEMCSCPadDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<GEMDetId, GEMCSCPadDigi> GEMCSCPadDigiCollection;

// Definition of a coincidence pad digi object
typedef std::pair<GEMCSCPadDigi,GEMCSCPadDigi> GEMCSCCoPadDigi;
// GEMDetId is detId of pad in 1st layer
typedef MuonDigiCollection<GEMDetId, GEMCSCCoPadDigi> GEMCSCCoPadDigiCollection;

#endif

