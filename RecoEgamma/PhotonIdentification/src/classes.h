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
    typedef VersionedSelector<edm::Ptr<reco::Photon> > VersionedPhotonSelector;
    typedef MakeVersionedSelector<reco::Photon> MakeVersionedPhotonSelector;
    typedef MakePtrFromCollection<reco::PhotonCollection> MakePhoPtrFromCollection;
    typedef PrintVIDToString<reco::Photon> PrintPhotonVIDToString;

    typedef VersionedSelector<edm::Ptr<pat::Photon> > VersionedPatPhotonSelector;
    typedef MakeVersionedSelector<pat::Photon> MakeVersionedPatPhotonSelector;
    typedef MakePtrFromCollection<std::vector<pat::Photon> > MakePatPtrFromCollection;
    typedef PrintVIDToString<pat::Photon> PrintPatPhotonVIDToString;

    //for using the selectors in python
    VersionedPhotonSelector vGsfElectronSelector;    
    MakeVersionedPhotonSelector vMakePhotonVersionedSelector;
    PrintPhotonVIDToString vPhoPrintVIDToString;
    MakePhoPtrFromCollection vPhoMakePtrFromCollection;  
    
    VersionedPatPhotonSelector vPatPhotonSelector; 
    MakeVersionedPatPhotonSelector vMakePatPhotonVersionedSelector;
    PrintPatPhotonVIDToString vPatPrintVIDToString;
    MakePatPtrFromCollection vPatMakePtrFromCollection;
    MakePtrFromCollection<std::vector<pat::Photon>, pat::Photon, reco::Photon> vPatToPhoMakePtrFromCollection;
    
  };
}
