#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/Parser/interface/SingleObjectSelector.h"
#include "PhysicsTools/Utilities/interface/PtMinSelector.h"
#include "PhysicsTools/Utilities/interface/MassRangeSelector.h"
#include "PhysicsTools/Utilities/interface/ChargeSelector.h"
#include "PhysicsTools/Utilities/interface/AndSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "PhysicsTools/UtilAlgos/interface/Merger.h"
#include "PhysicsTools/CandAlgos/src/CandCombiner.h"
#include "PhysicsTools/CandAlgos/src/CandShallowCloneCombiner.h"
#include "PhysicsTools/CandAlgos/src/CandReducer.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/Parser/interface/SingleObjectSelector.h"
#include "PhysicsTools/CandAlgos/src/TrivialDeltaRMatcher.h"
#include "PhysicsTools/CandAlgos/interface/ShallowCloneProducer.h"

DEFINE_SEAL_MODULE();

namespace reco {
  namespace modules {

    /// merge an arbitrary number of candidate collections
    typedef Merger<reco::CandidateCollection> CandMerger;

    /// configurable candidate selector
    typedef ObjectSelector<
              SingleElementCollectionSelector<
                reco::CandidateCollection,
                SingleObjectSelector<reco::Candidate>
              >
            > CandSelector;

    /// pt min candidate selector
    typedef ObjectSelector<
              SingleElementCollectionSelector<
                reco::CandidateCollection,
                PtMinSelector<reco::Candidate>
              >
            > PtMinCandSelector;

    /// configurable candidate combiner
    typedef ::CandCombiner<
              SingleObjectSelector<reco::Candidate>
            > CandCombiner;

    /// mass range and charge candidate selector
    typedef ::CandCombiner<
              AndSelector<
                ChargeSelector<reco::Candidate>,
                MassRangeSelector<reco::Candidate>
              >
           > MassRangeAndChargeCandCombiner;

    /// configurable candidate combiner
    typedef ::CandShallowCloneCombiner<
              SingleObjectSelector<reco::Candidate>
            > CandShallowCloneCombiner;

    /// mass range and charge candidate selector
    typedef ::CandShallowCloneCombiner<
              AndSelector<
                ChargeSelector<reco::Candidate>,
                MassRangeSelector<reco::Candidate>
              >
           > MassRangeAndChargeCandShallowCloneCombiner;

    typedef ShallowCloneProducer<
              reco::CandidateCollection
            > CandShallowCloneProducer;

DEFINE_ANOTHER_FWK_MODULE( CandSelector );
DEFINE_ANOTHER_FWK_MODULE( PtMinCandSelector );

DEFINE_ANOTHER_FWK_MODULE( MassRangeAndChargeCandCombiner );
DEFINE_ANOTHER_FWK_MODULE( CandCombiner );

DEFINE_ANOTHER_FWK_MODULE( CandShallowCloneCombiner );
DEFINE_ANOTHER_FWK_MODULE( MassRangeAndChargeCandShallowCloneCombiner );

DEFINE_ANOTHER_FWK_MODULE( CandReducer );
DEFINE_ANOTHER_FWK_MODULE( CandMerger );
DEFINE_ANOTHER_FWK_MODULE( CandShallowCloneProducer );

DEFINE_ANOTHER_FWK_MODULE( TrivialDeltaRMatcher );
  }
}
