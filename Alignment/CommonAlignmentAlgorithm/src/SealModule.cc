
#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//-----------------------------------------------------------------------------

#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentProducer.h"

//-----------------------------------------------------------------------------

#include "PhysicsTools/RecoAlgos/interface/TrackSelector.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "Alignment/CommonAlignmentAlgorithm/interface/AlignmentTrackSelector.h"

struct TrackConfigSelector {

  explicit TrackConfigSelector( const edm::ParameterSet & cfg ) :
    theSelector(cfg) {}

  bool operator()( const reco::Track & trk ) const {
    return theSelector(trk);
  }

  AlignmentTrackSelector theSelector;
};

typedef TrackSelector<TrackConfigSelector> AlignmentTrackSelectorModule;

//-----------------------------------------------------------------------------

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( AlignmentProducer );
DEFINE_ANOTHER_FWK_MODULE( AlignmentTrackSelectorModule );
