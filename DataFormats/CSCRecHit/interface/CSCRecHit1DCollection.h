#ifndef DataFormats_CSCRecHit1DCollection_H
#define DataFormats_CSCRecHit1DCollection_H


/** \class CSCRecHit1DCollection
 *
 * The collection of CSCRecHit1D's. See \ref CSCRecHit1DCollection.h for details.
 *
 */
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCRecHit1D.h>

#include <DataFormats/Common/interface/RangeMap.h>
#include <DataFormats/Common/interface/ClonePolicy.h>
#include <DataFormats/Common/interface/OwnVector.h>

typedef edm::RangeMap <CSCDetId, edm::OwnVector<CSCRecHit1D> > CSCRecHit1DCollection;

#endif
