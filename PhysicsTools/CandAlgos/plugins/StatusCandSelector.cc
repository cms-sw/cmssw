/* \class StatusCandSelector
 * 
 * Candidate Selector based on the status
 * Usage:
 * 
 * module leptons = StatusCandSelector {
 *   InputTag src = myCollection
 *   vint32 status =  { 1 }
 * };
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/StatusSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef SingleObjectSelector<
          reco::CandidateCollection,
          StatusSelector
        > StatusCandSelector;

DEFINE_FWK_MODULE( StatusCandSelector );
