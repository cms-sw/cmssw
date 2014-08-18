#ifndef EgammaAnalysis_ElectronTools_interface_VersionedGsfElectronSelector_h
#define EgammaAnalysis_ElectronTools_interface_VersionedGsfElectronSelector_h

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"

#include "PhysicsTools/SelectorUtils/interface/VersionedSelector.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include <iostream>

typedef VersionedSelector<reco::GsfElectronRef> VersionedGsfElectronSelector;

#endif
