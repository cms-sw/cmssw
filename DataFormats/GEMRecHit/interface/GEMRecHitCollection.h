#ifndef DataFormats_GEMRecHitCollection_H
#define DataFormats_GEMRecHitCollection_H

/** \class GEMRecHitCollection
 *  Collection of GEMRecHit for storage in the event
 *
 *  \author M. Maggi - INFN Bari
 */

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "DataFormats/Common/interface/IdToHitRange.h"

using GEMRecHitCollection = edm::IdToHitRange<GEMDetId, GEMRecHit>;
;

#endif
