/* \class reco::modules::CandSelector
 * 
 * Configurable Candidate Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/src/CandCombiner.h"
#include "PhysicsTools/Utilities/interface/AndSelector.h"
#include "PhysicsTools/Utilities/interface/MassRangeSelector.h"
#include "PhysicsTools/Utilities/interface/ChargeSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef CandCombiner<
          AndSelector<
            ChargeSelector<reco::Candidate>,
            MassRangeSelector<reco::Candidate>
          >
        > MassRangeAndChargeCandCombiner;

DEFINE_FWK_MODULE( MassRangeAndChargeCandCombiner );
