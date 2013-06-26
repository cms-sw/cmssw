#ifndef GEMDigi_GEMDigiCollection_h
#define GEMDigi_GEMDigiCollection_h
/** \class GEMDigiCollection
 *  
 *  \author Vadim Khotilovich
 *  \version $Id: GEMDigiCollection.h,v 1.1 2012/12/08 01:45:22 khotilov Exp $
 *  \date 21 Apr 2005
 */

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/GEMDigi/interface/GEMDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<GEMDetId, GEMDigi> GEMDigiCollection;

#endif

