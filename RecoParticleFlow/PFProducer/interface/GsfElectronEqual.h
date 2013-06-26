#ifndef RecoParticleFlow_PFProducer_GsfElectronEqual
#define RecoParticleFlow_PFProducer_GsfElectronEqual

#include "DataFormats/EgammaCandidates/interface/GsfElectron.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrack.h"
#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"


class GsfElectronEqual {
 public:
  GsfElectronEqual(const reco::GsfTrackRef& gsfRef):ref_(gsfRef) {;}
    ~GsfElectronEqual(){;}
    inline bool operator() (const reco::GsfElectron & gsfelectron) {
      return (gsfelectron.gsfTrack()==ref_);
    }
 private:
    reco::GsfTrackRef ref_;
};

#endif


