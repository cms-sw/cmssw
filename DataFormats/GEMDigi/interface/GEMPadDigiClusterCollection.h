#ifndef DataFormats_GEMDigi_GEMPadDigiClusterCollection_h
#define DataFormats_GEMDigi_GEMPadDigiClusterCollection_h

/** \class GEMPadDigiClusterCollection
 *  
 *  \author SVen Dildick
 */

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/GEMDigi/interface/GEMPadDigiCluster.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<GEMDetId, GEMPadDigiCluster> GEMPadDigiClusterCollection;

#endif

