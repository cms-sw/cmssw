#ifndef RecoParticleFlow_PFProducer_PFElectronExtraEqual
#define RecoParticleFlow_PFProducer_PFElectronExtraEqual

#include "DataFormats/GsfTrackReco/interface/GsfTrackFwd.h"
#include "DataFormats/TrackReco/interface/TrackFwd.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidateElectronExtra.h"

class PFElectronExtraEqual {
 public:
  PFElectronExtraEqual(const reco::GsfTrackRef & gsfTrackRef):ref_(gsfTrackRef) {;}
    ~PFElectronExtraEqual(){;}
    inline bool operator() (const reco::PFCandidateElectronExtra & extra) {
      return (ref_==extra.gsfTrackRef());
    }
 private:
    reco::GsfTrackRef ref_;
};

class PFElectronExtraKfEqual {
 public:
  PFElectronExtraKfEqual(const reco::TrackRef & trackRef):ref_(trackRef) {;}
    ~PFElectronExtraKfEqual(){;}
    inline bool operator() (const reco::PFCandidateElectronExtra & extra) {
      return (ref_==extra.kfTrackRef());
    }
 private:
    reco::TrackRef ref_;
};


#endif
