/* \class PtMinMuonSelector
 *
 * selects muon above a minumum pt cut
 *
 * \author: Luca Lista, INFN
 *
 */
#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"
#include "DataFormats/MuonReco/interface/Muon.h"

 typedef ObjectSelector<
           SingleElementCollectionSelector<
             reco::MuonCollection, 
             PtMinSelector<reco::Muon> 
           > 
         > PtMinMuonSelector;

DEFINE_FWK_MODULE( PtMinMuonSelector );
