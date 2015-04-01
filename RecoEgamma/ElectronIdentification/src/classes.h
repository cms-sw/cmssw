#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "RecoEgamma/ElectronIdentification/interface/VersionedGsfElectronSelector.h"
#include "RecoEgamma/ElectronIdentification/interface/VersionedPatElectronSelector.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"

#include "PhysicsTools/SelectorUtils/interface/MakePyVIDClassBuilder.h"
#include "PhysicsTools/SelectorUtils/interface/MakePtrFromCollection.h"
#include "PhysicsTools/SelectorUtils/interface/PrintVIDToString.h"

namespace RecoEgamma_ElectronIdentification {
  struct dictionary {    
    typedef MakeVersionedSelector<reco::GsfElectron> MakeVersionedGsfElectronSelector;
    typedef MakePtrFromCollection<reco::GsfElectronCollection> MakeGsfPtrFromCollection;
    typedef PrintVIDToString<reco::GsfElectron> PrintGsfElectronVIDToString;

    typedef MakeVersionedSelector<pat::Electron> MakeVersionedPatElectronSelector;
    typedef MakePtrFromCollection<std::vector<pat::Electron> > MakePatPtrFromCollection;
    typedef PrintVIDToString<pat::Electron> PrintPatElectronVIDToString;

    //for using the selectors in python
    VersionedGsfElectronSelector vGsfElectronSelector;    
    MakeVersionedGsfElectronSelector vMakeGsfElectronVersionedSelector;
    PrintGsfElectronVIDToString vGsfPrintVIDToString;
    MakeGsfPtrFromCollection vGsfMakePtrFromCollection;  
    
    VersionedPatElectronSelector vPatElectronSelector; 
    MakeVersionedPatElectronSelector vMakePatElectronVersionedSelector;
    PrintPatElectronVIDToString vPatPrintVIDToString;
    MakePatPtrFromCollection vPatMakePtrFromCollection;
    MakePtrFromCollection<std::vector<pat::Electron>, pat::Electron, reco::GsfElectron > vPatToGsfMakePtrFromCollection;;
    
  };
}


