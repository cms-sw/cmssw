#ifndef EgAmbiguityTools_H
#define EgAmbiguityTools_H

#include "DataFormats/EgammaCandidates/interface/GsfElectronFwd.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"
#include "DataFormats/CaloRecHit/interface/CaloClusterFwd.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/EcalRecHit/interface/EcalRecHitCollections.h"

namespace EgAmbiguityTools {
  // for clusters
  float sharedEnergy(reco::CaloCluster const& clu1,
                     reco::CaloCluster const& clu2,
                     EcalRecHitCollection const& barrelRecHits,
                     EcalRecHitCollection const& endcapRecHits);
  float sharedEnergy(reco::SuperClusterRef const& sc1,
                     reco::SuperClusterRef const& sc2,
                     EcalRecHitCollection const& barrelRecHits,
                     EcalRecHitCollection const& endcapRecHits);

  // for tracks
  int sharedHits(reco::GsfTrackRef const&, reco::GsfTrackRef const&);
  int sharedDets(reco::GsfTrackRef const&, reco::GsfTrackRef const&);

  // electrons comparison
  bool isBetter(reco::GsfElectron const&, reco::GsfElectron const&);
  bool isInnerMost(reco::GsfElectron const&, reco::GsfElectron const&);

}  // namespace EgAmbiguityTools

#endif
