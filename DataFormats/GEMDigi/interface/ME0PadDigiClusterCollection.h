#ifndef DataFormats_GEMDigi_ME0PadDigiClusterCollection_h
#define DataFormats_GEMDigi_ME0PadDigiClusterCollection_h

/** \class ME0PadDigiClusterCollection
 *  
 *  \author Sven Dildick
 */

#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/GEMDigi/interface/ME0PadDigiCluster.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<ME0DetId, ME0PadDigiCluster> ME0PadDigiClusterCollection;

#endif

