#ifndef DataFormats_RPCRecHitCollection_H
#define DataFormats_RPCRecHitCollection_H

/** \class RPCRecHitCollection
 *  Collection of RPCRecHit for storage in the event
 *
 *  \author M. Maggi - INFN Bari
 */

#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/Common/interface/IdToHitRange.h"

using RPCRecHitCollection = edm::IdToHitRange<RPCDetId, RPCRecHit>;

#endif
