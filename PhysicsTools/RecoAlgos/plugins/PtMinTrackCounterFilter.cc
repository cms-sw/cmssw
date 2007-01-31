#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "DataFormats/TrackReco/interface/Track.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"

typedef ObjectCountFilter<
          reco::TrackCollection, 
          PtMinSelector<reco::Track>
        > PtMinTrackCountFilter;

DEFINE_FWK_MODULE( PtMinTrackCountFilter );
