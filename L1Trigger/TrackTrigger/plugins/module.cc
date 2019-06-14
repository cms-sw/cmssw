/*! \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

/// The Builders

#include "L1Trigger/TrackTrigger/plugins/TTClusterBuilder.h"
typedef TTClusterBuilder<Ref_Phase2TrackerDigi_> TTClusterBuilder_Phase2TrackerDigi_;
DEFINE_FWK_MODULE(TTClusterBuilder_Phase2TrackerDigi_);

#include "L1Trigger/TrackTrigger/plugins/TTStubBuilder.h"
typedef TTStubBuilder<Ref_Phase2TrackerDigi_> TTStubBuilder_Phase2TrackerDigi_;
DEFINE_FWK_MODULE(TTStubBuilder_Phase2TrackerDigi_);

/// The Stub Finding Algorithms

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_official.h"
typedef ES_TTStubAlgorithm_official<Ref_Phase2TrackerDigi_> TTStubAlgorithm_official_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(TTStubAlgorithm_official_Phase2TrackerDigi_);

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_cbc3.h"
typedef ES_TTStubAlgorithm_cbc3<Ref_Phase2TrackerDigi_> TTStubAlgorithm_cbc3_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(TTStubAlgorithm_cbc3_Phase2TrackerDigi_);

/// The Clustering Algorithms

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_official.h"
typedef ES_TTClusterAlgorithm_official<Ref_Phase2TrackerDigi_> TTClusterAlgorithm_official_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(TTClusterAlgorithm_official_Phase2TrackerDigi_);

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_neighbor.h"
typedef ES_TTClusterAlgorithm_neighbor<Ref_Phase2TrackerDigi_> TTClusterAlgorithm_neighbor_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE(TTClusterAlgorithm_neighbor_Phase2TrackerDigi_);
