/* \class CandViewShallowCloneCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "CommonTools/CandAlgos/interface/CandCombiner.h"
#include "DataFormats/Candidate/interface/Candidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidate.h"
#include "DataFormats/Candidate/interface/NamedCompositeCandidateFwd.h"

namespace reco {
  namespace modules {
    typedef CandCombiner<StringCutObjectSelector<reco::Candidate, true>,
                         AnyPairSelector,
                         combiner::helpers::ShallowClone,
                         reco::NamedCompositeCandidateCollection>
        NamedCandViewShallowCloneCombiner;

    DEFINE_FWK_MODULE(NamedCandViewShallowCloneCombiner);

  }  // namespace modules
}  // namespace reco
