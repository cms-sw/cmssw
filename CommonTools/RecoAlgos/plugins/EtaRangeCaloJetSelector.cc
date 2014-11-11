/* \class EtaRangeCaloJetSelector
 *
 * selects calo-jet in the eta range
 *
 * \author: Silvio Donato
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/EtaRangeSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/JetReco/interface/CaloJet.h"

 typedef SingleObjectSelector<
           reco::CaloJetCollection, 
           EtaRangeSelector
         > EtaRangeCaloJetSelector;

DEFINE_FWK_MODULE( EtaRangeCaloJetSelector );
