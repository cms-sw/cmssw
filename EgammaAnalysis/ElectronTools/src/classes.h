#include "DataFormats/Common/interface/Wrapper.h"

//Add includes for your classes here
#include "EgammaAnalysis/ElectronTools/interface/VersionedGsfElectronSelector.h"
#include "EgammaAnalysis/ElectronTools/interface/VersionedPatElectronSelector.h"

namespace EgammaAnalysis_ElectronTools {
  struct dictionary {    
    //for using the selectors in python
    VersionedGsfElectronSelector vGsfElectronSelector; 
    VersionedPatElectronSelector vPatElectronSelector; 
  };
}
