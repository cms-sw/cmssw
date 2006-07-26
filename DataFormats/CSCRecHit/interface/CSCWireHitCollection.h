#ifndef DataFormats_CSCWireHitCollection_H
#define DataFormats_CSCWireHitCollection_H


/** \class CSCWireHitCollection
 *
 * The collection of CSCWireHit's. 
 *
 * \author Dominique Fortin - UCR
 */
#include <DataFormats/MuonDetId/interface/CSCDetId.h>
#include <DataFormats/CSCRecHit/interface/CSCWireHit.h>

#include <DataFormats/Common/interface/RangeMap.h>
#include <DataFormats/Common/interface/ClonePolicy.h>
#include <DataFormats/Common/interface/OwnVector.h>

typedef edm::RangeMap <CSCDetId, edm::OwnVector<CSCWireHit> > CSCWireHitCollection;

#endif
