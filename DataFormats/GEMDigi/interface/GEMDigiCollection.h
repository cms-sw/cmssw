#ifndef DataFormats_GEMDigi_GEMDigiCollection_h
#define DataFormats_GEMDigi_GEMDigiCollection_h

/** \class GEMDigiCollection
 *  
 *  \author Vadim Khotilovich
 */

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/GEMDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<GEMDetId, GEMDigi> GEMDigiCollection;

#endif

