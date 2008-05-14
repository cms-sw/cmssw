/* \class CandViewShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidateFwd.h"

namespace reco {
  namespace modules {
    typedef CandCombiner<
              StringCutObjectSelector<reco::Candidate>,
              AnyPairSelector,
              combiner::helpers::ShallowClonePtr,
              reco::NamedCompositeCandidateCollection 
            > NamedCandViewShallowCloneCombiner;

DEFINE_FWK_MODULE( NamedCandViewShallowCloneCombiner );

  }
}
