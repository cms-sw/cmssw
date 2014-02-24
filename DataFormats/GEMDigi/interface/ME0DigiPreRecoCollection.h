#ifndef GEMDigi_ME0DigiPreRecoCollection_h
#define GEMDigi_ME0DigiPreRecoCollection_h
/** \class ME0DigiPreRecoCollection
 *  
 *  \author Marcello Maggi
 *  \version $Id: ME0DigiPreRecoCollection.h,v 1.0  2014/02/02 10:09:22 mmaggi Exp $
 *  \date 2 February 2014
 */

#include <DataFormats/MuonDetId/interface/ME0DetId.h>
#include <DataFormats/GEMDigi/interface/ME0DigiPreReco.h>
#include <DataFormats/MuonData/interface/MuonDigiCollection.h>

typedef MuonDigiCollection<ME0DetId, ME0DigiPreReco> ME0DigiPreRecoCollection;

#endif

