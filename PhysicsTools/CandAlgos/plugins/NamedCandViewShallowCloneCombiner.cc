/* \class CandViewShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/CandAlgos/interface/NamedCandCombiner.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidateFwd.h"

namespace reco {
  namespace modules {
    typedef NamedCandCombiner<
              StringCutObjectSelector<reco::Candidate>,
              AnyPairSelector,
              combiner::helpers::ShallowClone
            > NamedCandViewShallowCloneCombiner;

DEFINE_FWK_MODULE( NamedCandViewShallowCloneCombiner );

  }
}
