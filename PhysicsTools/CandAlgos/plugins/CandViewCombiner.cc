/* \class reco::modules::plugins::CandViewCombiner
 * 
 * Configurable Candidate Selector reading
 * a View<Candidate> as input
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef reco::modules::CandCombiner<
                         StringCutObjectSelector<reco::Candidate>,
                         reco::CompositeCandidateCollection
                       > CandViewCombiner;
      
DEFINE_FWK_MODULE( CandViewCombiner );
