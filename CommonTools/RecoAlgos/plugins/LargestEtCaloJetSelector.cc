/* \class LargestEtCaloJetSelector
 *
 * selects the N calo-jets with largest Et
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/ObjectSelectorStream.h"
#include "CommonTools/UtilAlgos/interface/SortCollectionSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "CommonTools/Utils/interface/EtComparator.h"

typedef ObjectSelectorStream<
          SortCollectionSelector<
            reco::CaloJetCollection, 
              GreaterByEt<reco::CaloJet> 
          > 
        > LargestEtCaloJetSelector;

DEFINE_FWK_MODULE( LargestEtCaloJetSelector );
