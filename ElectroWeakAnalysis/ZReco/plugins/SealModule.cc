#include "PluginManager/ModuleDef.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "ElectroWeakAnalysis/ZReco/plugins/FiducialRegion.h"

#include "DataFormats/EgammaCandidates/interface/PixelMatchGsfElectronFwd.h"

#include "PhysicsTools/UtilAlgos/interface/ObjectSelector.h"
#include "PhysicsTools/UtilAlgos/interface/ObjectCountFilter.h"

typedef ObjectSelector<FiducialRegion> FiducialRegionSelector;


DEFINE_SEAL_MODULE();

DEFINE_ANOTHER_FWK_MODULE(FiducialRegionSelector);
