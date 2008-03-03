#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Calibration/Tools/interface/PhiRangeSelector.h"
#include "Calibration/Tools/interface/IMASelector.h"
#include "Calibration/Tools/plugins/SingleEleCalibSelector.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "PhysicsTools/RecoAlgos/interface/GsfElectronSelector.h"

//#include "Calibration/Tools/plugins/SelectorWithEventSetup.h"

DEFINE_SEAL_MODULE();


namespace reco {
  namespace modules {
    typedef SingleObjectSelector<reco::GsfElectronCollection,
                                 ::PhiRangeSelector> PhiRangeSelector; 
    DEFINE_ANOTHER_FWK_MODULE(PhiRangeSelector);

    typedef SingleObjectSelector<reco::GsfElectronCollection,
                                 ::IMASelector> IMASelector; 
    DEFINE_ANOTHER_FWK_MODULE(IMASelector);

  }
}


// typedef SelectorWithEventSetup<SingleEleCalibSelector> SingleElectronCalibrationSelector;
// DEFINE_ANOTHER_FWK_MODULE( SingleElectronCalibrationSelector );
