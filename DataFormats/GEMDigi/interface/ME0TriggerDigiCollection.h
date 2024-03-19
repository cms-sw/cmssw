#ifndef ME0TriggerDigi_ME0TriggerDigiCollection_h
#define ME0TriggerDigi_ME0TriggerDigiCollection_h

/** \class ME0TriggerDigiCollection
 *
 *  \author Sven Dildick (TAMU)
 *
 */

#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/ME0TriggerDigi.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<ME0DetId, ME0TriggerDigi> ME0TriggerDigiCollection;
typedef MuonDigiCollection<GEMDetId, ME0TriggerDigi> GE0TriggerDigiCollection;

#endif
