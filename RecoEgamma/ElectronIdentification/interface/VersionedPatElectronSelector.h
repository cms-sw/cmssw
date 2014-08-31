#ifndef EgammaAnalysis_ElectronTools_interface_VersionedPatElectronSelector_h
#define EgammaAnalysis_ElectronTools_interface_VersionedPatElectronSelector_h

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"

typedef VersionedSelector<edm::Ptr<pat::Electron> > VersionedPatElectronSelector;

#endif
