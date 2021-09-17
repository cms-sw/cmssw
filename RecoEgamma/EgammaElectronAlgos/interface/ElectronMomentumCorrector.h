#ifndef ElectronMomentumCorrector_H
#define ElectronMomentumCorrector_H

//===================================================================
// Author: Federico Ferri - INFN Milano, Bicocca university
//         Ivica Puljak - FESB, Split
// 12/2005
//adapted to CMSSW by U.Berthon, dec 2006
//===================================================================

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "TrackingTools/TrajectoryState/interface/TrajectoryStateOnSurface.h"
#include "DataFormats/Math/interface/LorentzVector.h"

namespace egamma {

  struct ElectronMomentum {
    const math::XYZTLorentzVector momentum;
    const float trackError;
    const float finalError;
  };

  ElectronMomentum correctElectronMomentum(reco::GsfElectron const&, TrajectoryStateOnSurface const&);
}  // namespace egamma

#endif
