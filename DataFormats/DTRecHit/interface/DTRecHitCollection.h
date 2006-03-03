#ifndef DataFormats_DTRecHitCollection_H
#define DataFormats_DTRecHitCollection_H

/** \class DTRecHitCollection
 *  Collection of 1DDTRecHitPair for storage in the event
 *
 *  $Date: 2006/02/15 10:14:22 $
 *  $Revision: 1.1 $
 *  \author G. Cerminara - INFN Torino
 */


#include "DataFormats/MuonDetId/interface/DTLayerId.h"
#include "DataFormats/DTRecHit/interface/DTRecHit1DPair.h"
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include <functional>

typedef edm::RangeMap <DTLayerId,
		       edm::OwnVector<DTRecHit1DPair,edm::ClonePolicy<DTRecHit1DPair> >,
		       edm::ClonePolicy<DTRecHit1DPair> > DTRecHitCollection;



/** \class DTSuperLayerIdComparator
 *  Comparator to retrieve rechits by SL.
 *
 *  $Date: $
 *  $Revision: $
 *  \author G. Cerminara - INFN Torino
 */

class DTSuperLayerIdComparator : public std::binary_function<DTSuperLayerId, DTSuperLayerId, bool> {
public:
  /// Constructor
  DTSuperLayerIdComparator() {}

  /// Destructor
  virtual ~DTSuperLayerIdComparator() {}

  // Operations
  bool operator()(const DTSuperLayerId& slId1, const DTSuperLayerId& slId2) const {
    return slId1 == slId2;
  }
protected:

private:

};
#endif




