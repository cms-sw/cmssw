#ifndef GEMDigi_GEMDigiCollection_h
#define GEMDigi_GEMDigiCollection_h
/** \class GEMDigiCollection
 *  
 *  \author Vadim Khotilovich
 *  \version $Id: GEMDigiCollection.h,v 1.2 2006/05/16 14:36:29 mmaggi Exp $
 *  \date 21 Apr 2005
 */

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/GEMDigi/interface/GEMDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<GEMDetId, GEMDigi> GEMDigiCollection;

#endif

