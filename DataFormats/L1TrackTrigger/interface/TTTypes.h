/*! \brief   Definition of all the relevant data types
 *
 *  \author Andrew W. Rose
 *  \author Nicola Pozzobon
 *  \date   2013, Jul 12
 *
 */

#ifndef L1_TRACK_TRIGGER_TYPES_H
#define L1_TRACK_TRIGGER_TYPES_H

/// Standard CMS Formats
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/Ptr.h"
#include "DataFormats/Common/interface/DetSetVector.h"
#include "DataFormats/Phase2TrackerDigi/interface/Phase2TrackerDigi.h"
#include "DataFormats/L1TrackTrigger/interface/TTCluster.h"
#include "DataFormats/L1TrackTrigger/interface/TTStub.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack.h"
#include "DataFormats/L1TrackTrigger/interface/TTTrack_TrackWord.h"

/// The reference types
typedef edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi> Ref_Phase2TrackerDigi_;

typedef edmNew::DetSetVector<TTCluster<Ref_Phase2TrackerDigi_> > TTClusterDetSetVec;
typedef edmNew::DetSetVector<TTStub<Ref_Phase2TrackerDigi_> > TTStubDetSetVec;

typedef edm::Ref<TTStubDetSetVec, TTStub<Ref_Phase2TrackerDigi_> > TTStubRef;
typedef edm::Ref<TTClusterDetSetVec, TTCluster<Ref_Phase2TrackerDigi_> > TTClusterRef;

typedef edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_> > TTStubDetSet;

#endif
