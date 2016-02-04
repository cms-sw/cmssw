#ifndef RecoParticleFlow_PFProducer_PhotonEqual
#define RecoParticleFlow_PFProducer_PhotonEqual

#include "DataFormats/EgammaCandidates/interface/Photon.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/EgammaReco/interface/SuperClusterFwd.h"


class PhotonEqual {
 public:
  PhotonEqual(const reco::SuperClusterRef& scRef):ref_(scRef) {;}
    ~PhotonEqual(){;}
    inline bool operator() (const reco::Photon & photon) {
      return (photon.superCluster()==ref_);
    }
 private:
    reco::SuperClusterRef ref_;
};

#endif


