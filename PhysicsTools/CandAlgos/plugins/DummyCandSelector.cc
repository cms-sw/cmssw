/* \class DummyCandSelector
 * 
 * Dummy Candidate selector module
 * 
 * module allCands = DummyCandSelector {
 *   InputTag src = myCollection
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/DummySelector.h"
#include "PhysicsTools/UtilAlgos/interface/FreeFunctionSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          reco::CandidateView,
          DummySelector
        > DummyCandSelector;

typedef SingleObjectSelector<
          reco::CandidateView,
          FreeFunctionSelector<reco::Candidate, dummy::select>
        > DummyFunCandSelector;

DEFINE_FWK_MODULE( DummyCandSelector );
DEFINE_FWK_MODULE( DummyFunCandSelector );
