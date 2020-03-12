#include "FWCore/PluginManager/interface/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "Calibration/Tools/interface/PhiRangeSelector.h"
#include "Calibration/Tools/interface/IMASelector.h"
#include "Calibration/Tools/plugins/SingleEleCalibSelector.h"

#include "CommonTools/UtilAlgos/interface/ObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleObjectSelector.h"
#include "CommonTools/UtilAlgos/interface/SingleElementCollectionSelector.h"

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "CommonTools/RecoAlgos/interface/GsfElectronSelector.h"

//#include "Calibration/Tools/plugins/SelectorWithEventSetup.h"
#include "Calibration/Tools/plugins/ElectronSqPtTkIsolationProducer.h"

namespace reco {
  namespace modules {
    typedef SingleObjectSelector<reco::GsfElectronCollection, ::PhiRangeSelector> PhiRangeSelector;
    DEFINE_FWK_MODULE(PhiRangeSelector);

    typedef SingleObjectSelector<reco::GsfElectronCollection, ::IMASelector> IMASelector;
    DEFINE_FWK_MODULE(IMASelector);

  }  // namespace modules
}  // namespace reco

DEFINE_FWK_MODULE(ElectronSqPtTkIsolationProducer);

// typedef SelectorWithEventSetup<SingleEleCalibSelector> SingleElectronCalibrationSelector;
// DEFINE_FWK_MODULE( SingleElectronCalibrationSelector );
