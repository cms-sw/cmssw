#ifndef DataFormats_RPCRecHitCollection_H
#define DataFormats_RPCRecHitCollection_H

/** \class RPCRecHitCollection
 *  Collection of RPCRecHit for storage in the event
 *
 *  $Date: 2006/04/12 20:49:06 $
 *  $Revision: 1.1 $
 *  \author M. Maggi - INFN Bari
 */


#include "DataFormats/MuonDetId/interface/RPCDetId.h"
#include "DataFormats/RPCRecHit/interface/RPCRecHit.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <functional>

typedef edm::RangeMap <RPCDetId,
		       edm::OwnVector<RPCRecHit,edm::ClonePolicy<RPCRecHit> >,
		       edm::ClonePolicy<RPCRecHit> > RPCRecHitCollection;


#endif




