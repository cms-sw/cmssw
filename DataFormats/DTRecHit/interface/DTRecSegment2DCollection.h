#ifndef DTRecHit_DTRecSegment2DCollection_h
#define DTRecHit_DTRecSegment2DCollection_h

/** \class DTRecSegment2DCollection
 *
 * Collection of DTRecSegment2D
 *  
 * $Date: 2006/03/20 12:42:28 $
 * $Revision: 1.1 $
 * \author Stefano Lacaprara - INFN Legnaro <stefano.lacaprara@pd.infn.it>
 * \author Riccardo Bellan - INFN TO <riccardo.bellan@cern.ch>
 *
 */

/* Base Class Headers */
#include <functional>

/* Collaborating Class Declarations */
#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"
#include "DataFormats/DTRecHit/interface/DTSLRecSegment2D.h"
#include "DataFormats/MuonDetId/interface/DTSuperLayerId.h"

/* C++ Headers */

/* ====================================================================== */

typedef edm::RangeMap<DTSuperLayerId,
  edm::OwnVector<DTSLRecSegment2D,edm::ClonePolicy<DTSLRecSegment2D> >,
  edm::ClonePolicy<DTSLRecSegment2D> > DTRecSegment2DCollection;

#endif // DTRecHit_DTRecSegment2DCollection_h
