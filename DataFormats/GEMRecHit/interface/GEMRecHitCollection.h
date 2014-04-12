#ifndef DataFormats_GEMRecHitCollection_H
#define DataFormats_GEMRecHitCollection_H

/** \class GEMRecHitCollection
 *  Collection of GEMRecHit for storage in the event
 *
 *  \author M. Maggi - INFN Bari
 */


#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMRecHit/interface/GEMRecHit.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <functional>

typedef edm::RangeMap <GEMDetId,
		       edm::OwnVector<GEMRecHit,edm::ClonePolicy<GEMRecHit> >,
		       edm::ClonePolicy<GEMRecHit> > GEMRecHitCollection;


#endif




