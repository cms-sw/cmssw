/* \class reco::modules::CandSelector
 * 
 * Configurable Candidate Selector
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/CandAlgos/interface/CandCombiner.h"
#include "PhysicsTools/UtilAlgos/interface/AndSelector.h"
#include "PhysicsTools/UtilAlgos/interface/MassRangeSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ChargeSelector.h"
#include "DataFormats/Candidate/interface/Candidate.h"

typedef CandCombiner<
          AndSelector<
            ChargeSelector<reco::Candidate>,
            MassRangeSelector<reco::Candidate>
          >
        > MassRangeAndChargeCandCombiner;

DEFINE_FWK_MODULE( MassRangeAndChargeCandCombiner );
