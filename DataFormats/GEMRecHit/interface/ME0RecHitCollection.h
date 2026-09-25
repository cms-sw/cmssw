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
#include "DataFormats/Common/interface/IdToHitRange.h"
#include <functional>

using ME0RecHitCollection = edm::IdToHitRange<ME0DetId, ME0RecHit>;
#endif
