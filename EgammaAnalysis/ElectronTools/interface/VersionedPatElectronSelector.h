#ifndef EgammaAnalysis_ElectronTools_interface_VersionedPatElectronSelector_h
#define EgammaAnalysis_ElectronTools_interface_VersionedPatElectronSelector_h

#include "DataFormats/PatCandidates/interface/Electron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

typedef VersionedSelector<pat::ElectronRef> VersionedPatElectronSelector;

#endif
