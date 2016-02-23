/*! \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

/// The Builders

#include "L1Trigger/TrackTrigger/plugins/TTClusterBuilder.h"
typedef TTClusterBuilder< Ref_Phase2TrackerDigi_> TTClusterBuilder_Phase2TrackerDigi_;
DEFINE_FWK_MODULE( TTClusterBuilder_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/plugins/TTStubBuilder.h"
typedef TTStubBuilder< Ref_Phase2TrackerDigi_ > TTStubBuilder_Phase2TrackerDigi_;
DEFINE_FWK_MODULE( TTStubBuilder_Phase2TrackerDigi_ );

/// The Stub Finding Algorithms

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_a.h"
typedef ES_TTStubAlgorithm_a< Ref_Phase2TrackerDigi_ > TTStubAlgorithm_a_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_a_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_window.h"
typedef ES_TTStubAlgorithm_window< Ref_Phase2TrackerDigi_ > TTStubAlgorithm_window_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_window_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_globalgeometry.h"
typedef ES_TTStubAlgorithm_globalgeometry< Ref_Phase2TrackerDigi_ > TTStubAlgorithm_globalgeometry_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_globalgeometry_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_pixelray.h"
typedef ES_TTStubAlgorithm_pixelray< Ref_Phase2TrackerDigi_ > TTStubAlgorithm_pixelray_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_pixelray_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_window2012.h"
typedef ES_TTStubAlgorithm_window2012< Ref_Phase2TrackerDigi_ > TTStubAlgorithm_window2012_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_window2012_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_window2013.h"
typedef ES_TTStubAlgorithm_window2013< Ref_Phase2TrackerDigi_ > TTStubAlgorithm_window2013_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_window2013_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_tab2013.h"
typedef ES_TTStubAlgorithm_tab2013< Ref_Phase2TrackerDigi_ > TTStubAlgorithm_tab2013_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_tab2013_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_cbc3.h"
typedef ES_TTStubAlgorithm_cbc3< Ref_Phase2TrackerDigi_ > TTStubAlgorithm_cbc3_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_cbc3_Phase2TrackerDigi_ );

/// The Clustering Algorithms

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_a.h"
typedef ES_TTClusterAlgorithm_a< Ref_Phase2TrackerDigi_ > TTClusterAlgorithm_a_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTClusterAlgorithm_a_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_broadside.h"
typedef ES_TTClusterAlgorithm_broadside< Ref_Phase2TrackerDigi_ > TTClusterAlgorithm_broadside_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTClusterAlgorithm_broadside_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_2d.h"
typedef ES_TTClusterAlgorithm_2d< Ref_Phase2TrackerDigi_ > TTClusterAlgorithm_2d_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTClusterAlgorithm_2d_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_2d2013.h"
typedef ES_TTClusterAlgorithm_2d2013< Ref_Phase2TrackerDigi_ > TTClusterAlgorithm_2d2013_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTClusterAlgorithm_2d2013_Phase2TrackerDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_neighbor.h"
typedef ES_TTClusterAlgorithm_neighbor< Ref_Phase2TrackerDigi_ > TTClusterAlgorithm_neighbor_Phase2TrackerDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTClusterAlgorithm_neighbor_Phase2TrackerDigi_ );



