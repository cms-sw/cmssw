#ifndef DataFormats_GEMDigi_GEMPadDigiCollection_h
#define DataFormats_GEMDigi_GEMPadDigiCollection_h

/** \class GEMPadDigiCollection
 *  
 *  \author Vadim Khotilovich
 */

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMPadDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<GEMDetId, GEMPadDigi> GEMPadDigiCollection;

#endif
