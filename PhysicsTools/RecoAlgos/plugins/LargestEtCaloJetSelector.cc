/* \class LargestEtCaloJetSelector
 *
 * selects the N calo-jets with largest Et
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SortCollectionSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "PhysicsTools/Utilities/interface/EtComparator.h"

typedef ObjectSelector<
          SortCollectionSelector<
            reco::CaloJetCollection, 
              GreaterByEt<reco::CaloJet> 
          > 
        > LargestEtCaloJetSelector;

DEFINE_FWK_MODULE( LargestEtCaloJetSelector );
