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

/// Templated aliases
template <typename T>
using TTClusterDetSetVecT = edmNew::DetSetVector<TTCluster<T> >;
template <typename T>
using TTStubDetSetVecT = edmNew::DetSetVector<TTStub<T> >;

template <typename T>
using TTClusterRefT = edm::Ref<TTClusterDetSetVecT<T>, TTCluster<T> >;
template <typename T>
using TTStubRefT = edm::Ref<TTStubDetSetVecT<T>, TTStub<T> >;

template <typename T>
using TTTrackPtrT = edm::Ptr<TTTrack<T> >;

/// Specialized aliases
typedef edm::Ref<edm::DetSetVector<Phase2TrackerDigi>, Phase2TrackerDigi> Ref_Phase2TrackerDigi_;

typedef edmNew::DetSet<TTStub<Ref_Phase2TrackerDigi_> > TTStubDetSet;

typedef TTClusterDetSetVecT<Ref_Phase2TrackerDigi_> TTClusterDetSetVec;
typedef TTStubDetSetVecT<Ref_Phase2TrackerDigi_> TTStubDetSetVec;

typedef TTClusterRefT<Ref_Phase2TrackerDigi_> TTClusterRef;
typedef TTStubRefT<Ref_Phase2TrackerDigi_> TTStubRef;

typedef TTTrackPtrT<Ref_Phase2TrackerDigi_> TTTrackPtr;

#endif
