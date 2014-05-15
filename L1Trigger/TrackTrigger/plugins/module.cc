/*! \author Nicola Pozzobon
 *  \date   2013, Jul 18
 *
 */

#include "DataFormats/L1TrackTrigger/interface/TTTypes.h"

/// The Builders

#include "L1Trigger/TrackTrigger/plugins/TTClusterBuilder.h"
typedef TTClusterBuilder< Ref_PixelDigi_> TTClusterBuilder_PixelDigi_;
DEFINE_FWK_MODULE( TTClusterBuilder_PixelDigi_ );

#include "L1Trigger/TrackTrigger/plugins/TTStubBuilder.h"
typedef TTStubBuilder< Ref_PixelDigi_ > TTStubBuilder_PixelDigi_;
DEFINE_FWK_MODULE( TTStubBuilder_PixelDigi_ );

/// The Stub Finding Algorithms

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_a.h"
typedef ES_TTStubAlgorithm_a< Ref_PixelDigi_ > TTStubAlgorithm_a_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_a_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_window.h"
typedef ES_TTStubAlgorithm_window< Ref_PixelDigi_ > TTStubAlgorithm_window_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_window_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_globalgeometry.h"
typedef ES_TTStubAlgorithm_globalgeometry< Ref_PixelDigi_ > TTStubAlgorithm_globalgeometry_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_globalgeometry_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_pixelray.h"
typedef ES_TTStubAlgorithm_pixelray< Ref_PixelDigi_ > TTStubAlgorithm_pixelray_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_pixelray_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_window2012.h"
typedef ES_TTStubAlgorithm_window2012< Ref_PixelDigi_ > TTStubAlgorithm_window2012_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_window2012_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_window2013.h"
typedef ES_TTStubAlgorithm_window2013< Ref_PixelDigi_ > TTStubAlgorithm_window2013_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_window2013_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_tab2013.h"
typedef ES_TTStubAlgorithm_tab2013< Ref_PixelDigi_ > TTStubAlgorithm_tab2013_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_tab2013_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTStubAlgorithm_cbc3.h"
typedef ES_TTStubAlgorithm_cbc3< Ref_PixelDigi_ > TTStubAlgorithm_cbc3_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTStubAlgorithm_cbc3_PixelDigi_ );

/// The Clustering Algorithms

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_a.h"
typedef ES_TTClusterAlgorithm_a< Ref_PixelDigi_ > TTClusterAlgorithm_a_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTClusterAlgorithm_a_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_broadside.h"
typedef ES_TTClusterAlgorithm_broadside< Ref_PixelDigi_ > TTClusterAlgorithm_broadside_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTClusterAlgorithm_broadside_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_2d.h"
typedef ES_TTClusterAlgorithm_2d< Ref_PixelDigi_ > TTClusterAlgorithm_2d_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTClusterAlgorithm_2d_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_2d2013.h"
typedef ES_TTClusterAlgorithm_2d2013< Ref_PixelDigi_ > TTClusterAlgorithm_2d2013_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTClusterAlgorithm_2d2013_PixelDigi_ );

#include "L1Trigger/TrackTrigger/interface/TTClusterAlgorithm_neighbor.h"
typedef ES_TTClusterAlgorithm_neighbor< Ref_PixelDigi_ > TTClusterAlgorithm_neighbor_PixelDigi_;
DEFINE_FWK_EVENTSETUP_MODULE( TTClusterAlgorithm_neighbor_PixelDigi_ );



