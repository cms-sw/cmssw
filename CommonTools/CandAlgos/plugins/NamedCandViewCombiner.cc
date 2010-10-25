/* \class reco::modules::plugins::CandViewCombiner
 * 
 * Configurable Candidate Selector reading
 * a View<Candidate> as input
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

typedef reco::modules::CandCombiner<
                         StringCutObjectSelector<reco::Candidate, true>,
                         AnyPairSelector,
                         combiner::helpers::NormalClone,
                         reco::NamedCompositeCandidateCollection
                       > NamedCandViewCombiner;
      
DEFINE_FWK_MODULE(NamedCandViewCombiner);
