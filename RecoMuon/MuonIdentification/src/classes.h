#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "RecoMuon/MuonIdentification/interface/VersionedMuonSelector.h"
#include "PhysicsTools/SelectorUtils/interface/MakePyVIDClassBuilder.h"

namespace RecoMuon_MuonIdentification {
  struct dictionary {    
    //for using the selectors in python
    VersionedMuonSelector vMuonSelector; 
    MakeVersionedSelector<reco::Muon> vMakeMuonVersionedSelector;
  };
}
