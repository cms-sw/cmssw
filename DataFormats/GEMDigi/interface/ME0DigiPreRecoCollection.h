#ifndef DataFormats_GEMDigi_ME0DigiPreRecoCollection_h
#define DataFormats_GEMDigi_ME0DigiPreRecoCollection_h

/** \class ME0DigiPreRecoCollection
 *  
 *  \author Marcello Maggi
 */

#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/GEMDigi/interface/ME0DigiPreReco.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"

typedef MuonDigiCollection<ME0DetId, ME0DigiPreReco> ME0DigiPreRecoCollection;
typedef MuonDigiCollection<ME0DetId, int> ME0DigiPreRecoMap;

#endif
