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
#include "PhysicsTools/CandAlgos/src/CandReducer.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "PhysicsTools/Parser/interface/SingleObjectSelector.h"

namespace cand {
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

DEFINE_SEAL_MODULE();
DEFINE_ANOTHER_FWK_MODULE( CandSelector );
DEFINE_ANOTHER_FWK_MODULE( PtMinCandSelector );
DEFINE_ANOTHER_FWK_MODULE( MassRangeAndChargeCandCombiner );
DEFINE_ANOTHER_FWK_MODULE( CandCombiner );
DEFINE_ANOTHER_FWK_MODULE( CandReducer );
DEFINE_ANOTHER_FWK_MODULE( CandMerger );
  }
}
