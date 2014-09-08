#ifndef EgammaAnalysis_ElectronTools_interface_VersionedGsfElectronSelector_h
#define EgammaAnalysis_ElectronTools_interface_VersionedGsfElectronSelector_h

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"

typedef VersionedSelector<edm::Ptr<reco::GsfElectron> > VersionedGsfElectronSelector;

#endif
