#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "RecoEgamma/ElectronIdentification/interface/VersionedGsfElectronSelector.h"
#include "RecoEgamma/ElectronIdentification/interface/VersionedPatElectronSelector.h"

namespace RecoEgamma_ElectronIdentification {
  struct dictionary {    
    //for using the selectors in python
    VersionedGsfElectronSelector vGsfElectronSelector; 
    VersionedPatElectronSelector vPatElectronSelector; 
  };
}
