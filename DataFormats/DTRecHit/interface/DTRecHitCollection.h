#ifndef DataFormats_DTRecHitCollection_H
#define DataFormats_DTRecHitCollection_H

/** \class DTRecHitCollection
 *  Collection of 1DDTRecHitPair for storage in the event
 *
 *  $Date: 2006/03/03 11:28:26 $
 *  $Revision: 1.2 $
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
 *  $Date: 2006/03/03 11:28:26 $
 *  $Revision: 1.2 $
 *  \author G. Cerminara - INFN Torino
 */

class DTSuperLayerIdComparator { //: public std::binary_function<DTSuperLayerId, DTSuperLayerId, bool> {
public:
  // // Operations
  // bool operator()(const DTSuperLayerId& slId1, const DTSuperLayerId& slId2) const {
  //   return slId1 == slId2;
  // }

  bool operator()(const DTLayerId& l1, const DTLayerId& l2) const {
    if (l1.superlayerId() == l2.superlayerId()) return false;
    return (l1.superlayerId()<l2.superlayerId());
  }

protected:

private:

};
#endif




