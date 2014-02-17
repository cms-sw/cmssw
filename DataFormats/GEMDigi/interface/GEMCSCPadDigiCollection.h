#ifndef GEMDigi_GEMCSCPadDigiCollection_h
#define GEMDigi_GEMCSCPadDigiCollection_h
/** \class GEMCSCPadDigiCollection
 *  
 *  \author Vadim Khotilovich
 *  \version $Id: GEMCSCPadDigiCollection.h,v 1.1 2013/01/18 04:21:50 khotilov Exp $
 *  \date 21 Apr 2005
 */

#include <DataFormats/MuonDetId/interface/GEMDetId.h>
#include <DataFormats/GEMDigi/interface/GEMCSCPadDigi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<GEMDetId, GEMCSCPadDigi> GEMCSCPadDigiCollection;

#endif

