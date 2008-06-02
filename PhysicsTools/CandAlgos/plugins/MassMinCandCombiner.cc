/* \class MassMinCandCombiner
 * 
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "PhysicsTools/UtilAlgos/interface/AndSelector.h"
#include "PhysicsTools/UtilAlgos/interface/MassMinSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {
  namespace modules {
    typedef CandCombiner<
              reco::CandidateCollection,
              MassMinSelector
            > MassMinCandCombiner;

DEFINE_FWK_MODULE( MassMinCandCombiner );

  }
}
