#ifndef DataFormats_DTRecHitCollection_H
#define DataFormats_DTRecHitCollection_H

/** \class DTRecHitCollection
 *  Collection of 1DDTRecHitPair for storage in the event. See \ref DTRecHitCollection.h for details
 *
 *  $Date: 2006/06/26 13:11:47 $
 *  $Revision: 1.6 $
 *  \author G. Cerminara - INFN Torino
 */


#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <functional>

typedef edm::RangeMap <DTLayerId, edm::OwnVector<DTRecHit1DPair> > DTRecHitCollection;



#endif




