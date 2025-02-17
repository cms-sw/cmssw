#ifndef DataFormats_ME0StubCollection_H
#define DataFormats_ME0StubCollection_H

/** \class ME0StubCollection
 *
 * The collection of ME0Stub's. See \ref CSCSegmentCollection.h for details from which is derived.
 *
 *  \author Woohyeon Heo
 */

#include "DataFormats/MuonDetId/interface/GEMDetId.h"
#include "DataFormats/GEMDigi/interface/ME0Stub.h"

#include "DataFormats/Common/interface/RangeMap.h"
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/OwnVector.h"

typedef edm::RangeMap<GEMDetId, edm::OwnVector<ME0Stub> > ME0StubCollection;

#include "DataFormats/Common/interface/Ref.h"
typedef edm::Ref<ME0StubCollection> ME0StubRef;

#endif