#ifndef CSCRecHitD_CSCStripHitCollection_H
#define CSCRecHitD_CSCStripHitCollection_H


/** \class CSCStripHitCollection
 *
 * The collection of CSCStripHit's. 
 *
 */

#include <RecoLocalMuon/CSCRecHitD/src/CSCStripHit.h>

#include <DataFormats/MuonDetId/interface/CSCDetId.h>

#include <DataFormats/Common/interface/RangeMap.h>
#include <DataFormats/Common/interface/ClonePolicy.h>
#include <DataFormats/Common/interface/OwnVector.h>


typedef edm::RangeMap <CSCDetId, edm::OwnVector<CSCStripHit> > CSCStripHitCollection;

#endif
