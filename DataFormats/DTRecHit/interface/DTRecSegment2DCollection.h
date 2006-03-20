#ifndef DTRecHit_DTRecSegment2DCollection_h
#define DTRecHit_DTRecSegment2DCollection_h

/** \class DTRecSegment2DCollection
 *
 * Collection of DTRecSegment2D
 *  
 * $Date: 23/02/2006 13:08:28 CET $
 * $Revision: 1.0 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 *
 */

/* Base Class Headers */
#include <functional>

/* Collaborating Class Declarations */
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/MuonData/interface/MuonDigiCollection.h"
#include "DataFormats/DTRecHit/interface/DTRecSegment2D.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

/* C++ Headers */

/* ====================================================================== */

typedef edm::RangeMap<DTSuperLayerId,
  edm::OwnVector<DTRecSegment2D,edm::ClonePolicy<DTRecSegment2D> >,
  edm::ClonePolicy<DTRecSegment2D> > DTRecSegment2DCollection;

#endif // DTRecHit_DTRecSegment2DCollection_h
