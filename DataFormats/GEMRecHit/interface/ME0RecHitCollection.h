#ifndef DataFormats_ME0RecHitCollection_H
#define DataFormats_ME0RecHitCollection_H

/** \class ME0RecHitCollection
 *  Collection of ME0RecHit for storage in the event
 *
 *  $Date: 2013/04/24 16:54:23 $
 *  $Revision: 1.1 $
 *  \author M. Maggi - INFN Bari
 */


#include "DataFormats/MuonDetId/interface/ME0DetId.h"
#include "DataFormats/GEMRecHit/interface/ME0RecHit.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <functional>

typedef edm::RangeMap <ME0DetId,
		       edm::OwnVector<ME0RecHit,edm::ClonePolicy<ME0RecHit> >,
		       edm::ClonePolicy<ME0RecHit> > ME0RecHitCollection;


#endif




