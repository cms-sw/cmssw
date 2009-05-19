/** \class GsfElectronSelector
 */

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "RecoBTag/SoftLepton/interface/GenericSelectorByValueMap.h"

typedef GenericSelectorByValueMap<reco::GsfElectron> GsfElectronSelector;

//------------------------------------------------------------------------------

#include "FWCore/Framework/interface/MakerMacros.h"
DEFINE_FWK_MODULE(GsfElectronSelector);
