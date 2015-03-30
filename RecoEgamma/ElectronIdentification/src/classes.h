#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "RecoEgamma/ElectronIdentification/interface/VersionedGsfElectronSelector.h"
#include "RecoEgamma/ElectronIdentification/interface/VersionedPatElectronSelector.h"

#include "PhysicsTools/SelectorUtils/interface/MakePyVIDClassBuilder.h"

namespace RecoEgamma_ElectronIdentification {
  struct dictionary {    
    //for using the selectors in python
    VersionedGsfElectronSelector vGsfElectronSelector; 
    VersionedPatElectronSelector vPatElectronSelector; 
    MakeVersionedSelector<reco::GsfElectron> vMakeGsfElectronVersionedSelector;
    MakeVersionedSelector<pat::Electron> vMakePatElectronVersionedSelector;
    
  };
}


