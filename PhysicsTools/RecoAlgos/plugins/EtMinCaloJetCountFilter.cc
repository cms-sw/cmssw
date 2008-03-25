/* \class EtMinCaloJetCountFilter
 *
 * Filters events if at least N calo-jets above 
 * an Et cut are present
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/UtilAlgos/interface/EtMinSelector.h"

typedef ObjectCountFilter<
          reco::CaloJetCollection, 
          EtMinSelector
        > EtMinCaloJetCountFilter;

DEFINE_FWK_MODULE( EtMinCaloJetCountFilter );
