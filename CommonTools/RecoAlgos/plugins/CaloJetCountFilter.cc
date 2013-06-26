/* \class CaloJetCountFilter
 *
 * Filters events if at least N calo-jets
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "CommonTools/UtilAlgos/interface/ObjectCountFilter.h"

 typedef ObjectCountFilter<
           reco::CaloJetCollection
         >::type CaloJetCountFilter;

DEFINE_FWK_MODULE( CaloJetCountFilter );
