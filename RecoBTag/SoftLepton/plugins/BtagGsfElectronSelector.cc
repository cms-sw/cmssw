/** \class BtagGsfElectronSelector
 */

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoBTag/SoftLepton/interface/GenericSelectorByValueMap.h"

// "float" is the type stored in the ValueMap
typedef edm::GenericSelectorByValueMap<reco::GsfElectron, float> BtagGsfElectronSelector;

//------------------------------------------------------------------------------

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(BtagGsfElectronSelector);
