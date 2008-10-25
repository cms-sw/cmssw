#include "FWCore/Framework/interface/MakerMacros.h"
#include "PhysicsTools/UtilAlgos/interface/PtMinSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "DataFormats/JetReco/interface/PFJet.h"

 typedef SingleObjectSelector<
           reco::PFJetCollection, 
           PtMinSelector 
         > PtMinPFJetSelector;

DEFINE_FWK_MODULE( PtMinPFJetSelector );
