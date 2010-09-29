



#include "PhysicsTools/TagAndProbe/interface/GenericElectronSelection.h"
#include "FWCore/Framework/interface/MakerMacros.h"

//define this as a plug-in
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
typedef GenericElectronSelection< reco::GsfElectron > gsfElectronSelector;
DEFINE_FWK_MODULE( gsfElectronSelector );

#include "DataFormats/PatCandidates/interface/Electron.h"
typedef GenericElectronSelection< pat::Electron > patElectronSelector;
DEFINE_FWK_MODULE( patElectronSelector  );
