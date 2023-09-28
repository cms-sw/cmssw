#ifndef DataFormats_DTRecHitCollection_H
#define DataFormats_DTRecHitCollection_H

/** \class DTRecHitCollection
 *  Collection of 1DDTRecHitPair for storage in the event. See \ref DTRecHitCollection.h for details
 *
 *  \author G. Cerminara - INFN Torino
 */

#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/Common/interface/IdToHitRange.h"

typedef edm::IdToHitRange<DTLayerId, DTRecHit1DPair> DTRecHitCollection;

#endif
