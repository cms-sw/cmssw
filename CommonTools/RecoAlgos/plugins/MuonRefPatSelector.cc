#include "FWCore/Framework/interface/MakerMacros.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/StringCutObjectSelector.h"
#include "DataFormats/PatCandidates/interface/Muon.h"


 typedef SingleObjectSelector<
           pat::MuonCollection, 
           StringCutObjectSelector<pat::Muon>,
           pat::MuonRefVector
         > MuonRefPatSelector;

DEFINE_FWK_MODULE( MuonRefPatSelector );
