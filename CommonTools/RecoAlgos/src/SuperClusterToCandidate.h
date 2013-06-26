#ifndef RecoAlgos_SuperClusterToCandidate_h
#define RecoAlgos_SuperClusterToCandidate_h
#include "CommonTools/RecoAlgos/src/MassiveCandidateConverter.h"
#include "CommonTools/RecoAlgos/src/CandidateProducer.h"
#include "DataFormats/EgammaReco/interface/SuperCluster.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidate.h"
#include "DataFormats/RecoCandidate/interface/RecoEcalCandidateFwd.h"
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace converter {
  struct SuperClusterToCandidate : public MassiveCandidateConverter {
    typedef reco::SuperCluster value_type;
    typedef reco::SuperClusterCollection Components;
    typedef reco::RecoEcalCandidate Candidate;
    SuperClusterToCandidate(const edm::ParameterSet & cfg) : 
      MassiveCandidateConverter(cfg) {
    }
    void convert(reco::SuperClusterRef scRef, reco::RecoEcalCandidate & c) const {
      const reco::SuperCluster & sc = * scRef;
      math::XYZPoint v(0, 0, 0); // this should be taken from something else...
      math::XYZVector p = sc.energy() * (sc.position() - v).unit();
      double t = sqrt(massSqr_ + p.mag2());
      c.setCharge(0);
      c.setVertex(v);
      c.setP4(reco::Candidate::LorentzVector(p.x(), p.y(), p.z(), t));
      c.setSuperCluster(scRef);
      c.setPdgId(particle_.pdgId());
    }
  };

  namespace helper {
    template<>
    struct CandConverter<reco::SuperCluster> { 
      typedef SuperClusterToCandidate type;
    };
  }
}

#endif
