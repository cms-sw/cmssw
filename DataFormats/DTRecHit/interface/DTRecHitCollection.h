#ifndef DataFormats_DTRecHitCollection_H
#define DataFormats_DTRecHitCollection_H

/** \class DTRecHitCollection
 *  Collection of 1DDTRecHitPair for storage in the event
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */


#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"

typedef MuonDigiCollection<DTLayerId, DTRecHit1DPair> DTRecHitCollection;

#endif
