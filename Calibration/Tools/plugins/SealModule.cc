#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Calibration/Tools/interface/PhiRangeSelector.h"
#include "Calibration/Tools/interface/IMASelector.h"
#include "Calibration/Tools/interface/SingleEleCalibSelector.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"
#include "PhysicsTools/RecoAlgos/interface/PixelMatchGsfElectronSelector.h"

#include "Calibration/Tools/plugins/SelectorWithEventSetup.h"

DEFINE_SEAL_MODULE();


namespace reco {
  namespace modules {
    typedef SingleObjectSelector<reco::PixelMatchGsfElectronCollection,
                                 ::PhiRangeSelector> PhiRangeSelector; 
    DEFINE_ANOTHER_FWK_MODULE(PhiRangeSelector);

    typedef SingleObjectSelector<reco::PixelMatchGsfElectronCollection,
                                 ::IMASelector> IMASelector; 
    DEFINE_ANOTHER_FWK_MODULE(IMASelector);

  }
}


typedef SelectorWithEventSetup<SingleEleCalibSelector> SingleElectronCalibrationSelector;
DEFINE_ANOTHER_FWK_MODULE( SingleElectronCalibrationSelector );
