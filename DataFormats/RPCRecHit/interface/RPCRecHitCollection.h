#ifndef DataFormats_RPCRecHitCollection_H
#define DataFormats_RPCRecHitCollection_H

/** \class RPCRecHitCollection
 *  Collection of RPCRecHit for storage in the event
 *
 *  \author M. Maggi - INFN Bari
 */
#include <vector>

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include <functional>

using RPCRecHitCollection = edm::RangeMap<RPCDetId, std::vector<RPCRecHit>, edm::CopyPolicy<RPCRecHit>>;

#endif
