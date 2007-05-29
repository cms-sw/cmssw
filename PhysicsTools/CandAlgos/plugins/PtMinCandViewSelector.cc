/* \class PtMinCandViewSelector
 * 
 * Candidate Selector based on a minimum pt cut.
 * Reads a edm::View<Candidate> as input
 * and saves a vector of references
 * Usage:
 * 
 * module selectedCands = PtMinCandViewSelector {
 *   InputTag src = myCollection
 *   double ptMin = 15.0
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          edm::View<reco::Candidate>,
          PtMinSelector<reco::Candidate>
        > PtMinCandViewSelector;

DEFINE_FWK_MODULE( PtMinCandViewSelector );
