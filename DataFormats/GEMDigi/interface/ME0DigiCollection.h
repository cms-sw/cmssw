#ifndef ME0Digi_ME0DigiCollection_h
#define ME0Digi_ME0DigiCollection_h

/** \class ME0DigiCollection
 *  
 *  \author Sven Dildick
 */

#include <DataFormats/MuonDetId/interface/ME0DetId.h>
#include <DataFormats/GEMDigi/interface/ME0Digi.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<ME0DetId, ME0Digi> ME0DigiCollection;

#endif

