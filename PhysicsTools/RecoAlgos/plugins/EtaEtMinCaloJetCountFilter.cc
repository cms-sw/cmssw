/* \class EtaEtMinCaloJetCountFilter
 *
 * Filters events if at least N calo-jets above 
 * an Et cut and within an eta range are present
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "DataFormats/JetReco/interface/CaloJet.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"
#include "PhysicsTools/UtilAlgos/interface/AndSelector.h"
#include "PhysicsTools/UtilAlgos/interface/EtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/EtaRangeSelector.h"

 typedef ObjectCountFilter<
           reco::CaloJetCollection, 
	      AndSelector<
               EtMinSelector,
	       EtaRangeSelector
           > 
         > EtaEtMinCaloJetCountFilter;

DEFINE_FWK_MODULE( EtaEtMinCaloJetCountFilter );
