#ifndef CSCRecHitD_CSCWireHitCollection_H
#define CSCRecHitD_CSCWireHitCollection_H


/** \class CSCWireHitCollection
 *
 * The collection of CSCWireHit's. 
 *
 */

#include <RecoLocalMuon/CSCRecHitD/src/CSCWireHit.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <DataFormats/Common/interface/RangeMap.h>
#include <DataFormats/Common/interface/ClonePolicy.h>
#include <DataFormats/Common/interface/OwnVector.h>

typedef edm::RangeMap <CSCDetId, edm::OwnVector<CSCWireHit> > CSCWireHitCollection;

#endif
