#ifndef DataFormats_GEMRecHitCollection_H
#define DataFormats_GEMRecHitCollection_H

/** \class GEMRecHitCollection
 *  Collection of GEMRecHit for storage in the event
 *
 *  \author M. Maggi - INFN Bari
 */

#include <vector>
#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include <functional>

using GEMRecHitCollection = edm::RangeMap<GEMDetId, std::vector<GEMRecHit>, edm::CopyPolicy<GEMRecHit>>;
;

#endif
