/* \class MuonPairMassFilter
 *
 * Filters events if at least N muons pairs
 * with an invariant mass within a specified range
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"
#include "PhysicsTools/UtilAlgos/interface/MasslessInvariantMass.h"
#include "PhysicsTools/UtilAlgos/interface/RangeObjectPairSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectPairFilter.h"

typedef ObjectPairFilter<
          reco::MuonCollection,
          RangeObjectPairSelector<
            MasslessInvariantMass
          >
        > MuonPairMassFilter;

DEFINE_FWK_MODULE( MuonPairMassFilter );
