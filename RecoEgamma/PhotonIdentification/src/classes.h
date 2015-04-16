#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "DataFormats/PatCandidates/interface/Photon.h"
#include "DataFormats/EgammaCandidates/interface/PhotonFwd.h"

#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
#include "PhysicsTools/SelectorUtils/interface/MakePyVIDClassBuilder.h"
#include "PhysicsTools/SelectorUtils/interface/MakePtrFromCollection.h"
#include "PhysicsTools/SelectorUtils/interface/PrintVIDToString.h"

namespace RecoEgamma_PhotonIdentification {
  struct dictionary {    
    typedef VersionedSelector<reco::Photon> VersionedPhotonSelector;
    typedef MakeVersionedSelector<reco::Photon> MakeVersionedPhotonSelector;
    typedef MakePtrFromCollection<reco::Photon> MakePhoPtrFromCollection;
    typedef PrintVIDToString<reco::Photon> PrintPhotonVIDToString;

    typedef VersionedSelector<pat::Photon> VersionedPatPhotonSelector;
    typedef MakeVersionedSelector<pat::Electron> MakeVersionedPatElectronSelector;
    typedef MakePtrFromCollection<std::vector<pat::Electron> > MakePatPtrFromCollection;
    typedef PrintVIDToString<pat::Electron> PrintPatElectronVIDToString;

    //for using the selectors in python
    VersionedPhotonSelector vGsfElectronSelector;    
    MakeVersionedGsfElectronSelector vMakePhotonVersionedSelector;
    PrintPhotonVIDToString vPhoPrintVIDToString;
    MakePhoPtrFromCollection vPhoMakePtrFromCollection;  
    
    VersionedPatPhotonSelector vPatPhotonSelector; 
    MakeVersionedPatPhotonSelector vMakePatPhotonVersionedSelector;
    PrintPatPhotonVIDToString vPatPrintVIDToString;
    MakePatPtrFromCollection vPatMakePtrFromCollection;
    MakePtrFromCollection<std::vector<pat::Photon>, pat::Photon, reco::Photon > vPatToPhoMakePtrFromCollection;
    
  };
}
