#ifndef RecoAlgos_ObjectPairFilters_h
#define RecoAlgos_ObjectPairFilters_h
#include "PhysicsTools/UtilAlgos/interface/ObjectPairFilter.h"
#include "PhysicsTools/Utilities/interface/MasslessInvariantMass.h"
#include "DataFormats/MuonReco/interface/Muon.h"
#include "DataFormats/EgammaCandidates/interface/Electron.h"

namespace reco {
  namespace modules {
    /// filter on electron pairs based on invariant mass
    typedef ObjectPairFilter<
              reco::ElectronCollection,
              MasslessInvariantMass<reco::Electron> 
            > ElectronPairMassFilter;

    /// filter on muon pairs based on invariant mass
    typedef ObjectPairFilter<
              reco::MuonCollection,
              MasslessInvariantMass<reco::Muon> 
            > MuonPairMassFilter;
  }
}
#endif
