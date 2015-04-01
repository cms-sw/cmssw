#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "RecoMuon/MuonIdentification/interface/VersionedMuonSelector.h"
#include "DataFormats/MuonReco/interface/MuonFwd.h"

#include "DataFormats/PatCandidates/interface/Muon.h"

#include "PhysicsTools/SelectorUtils/interface/MakePyVIDClassBuilder.h"
#include "PhysicsTools/SelectorUtils/interface/MakePtrFromCollection.h"
#include "PhysicsTools/SelectorUtils/interface/PrintVIDToString.h"

namespace RecoMuon_MuonIdentification {
  struct dictionary {    
    //for using the selectors in python
    VersionedMuonSelector vMuonSelector; 
    
    MakeVersionedSelector<reco::Muon> vMakeMuonVersionedSelector;
    MakePtrFromCollection<reco::MuonCollection> vMakeMuonPtrFromCollection;
    MakePtrFromCollection<std::vector<pat::Muon>,pat::Muon,reco::Muon> vMakePatToRecoMuonPtrFromCollection;
    PrintVIDToString<reco::Muon> vPrintMuonVIDToString;
  };
}
