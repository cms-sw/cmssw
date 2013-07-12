#ifndef DTRecSegment4DCollection_H
#define DTRecSegment4DCollection_H

/** \class DTRecSegment4DCollection
 *  
 *  Collection of DTRecSegment4D. See \ref DTRecSegment4DCollection.h for details
 *
 *  $Date: 2006/06/29 17:18:27 $
 *  $Revision: 1.5 $
 *  \author R. Bellan - INFN Torino
 */

/* Base Class Headers */
#include <functional>

/* Collaborating Class Declarations */
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment4D.h"
#include "DataFormats/MuonDetId/interface/DTChamberId.h"


typedef edm::RangeMap<DTChamberId, edm::OwnVector<DTRecSegment4D> > DTRecSegment4DCollection;

#include "DataFormats/Common/interface/Ref.h"
typedef edm::Ref<DTRecSegment4DCollection> DTRecSegment4DRef;

#endif

