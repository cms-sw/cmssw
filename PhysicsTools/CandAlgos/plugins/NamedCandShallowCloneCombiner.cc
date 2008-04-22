/* \class CandShallowCloneCombiner
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
              reco::CandidateCollection,
              StringCutObjectSelector<reco::Candidate>,
              reco::NamedCompositeCandidateCollection,
              AnyPairSelector,
              combiner::helpers::ShallowClone
            > NamedCandShallowCloneCombiner;

DEFINE_FWK_MODULE( NamedCandShallowCloneCombiner );

  }
}
