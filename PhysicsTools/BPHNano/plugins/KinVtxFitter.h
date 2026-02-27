#ifndef PhysicsTools_BPHNano_KinVtxFitter
#define PhysicsTools_BPHNano_KinVtxFitter

#include <vector>

#include "CommonTools/Statistics/interface/ChiSquaredProbability.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicState.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicParticle.h"
#include "RecoVertex/KinematicFitPrimitives/interface/RefCountedKinematicVertex.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"

class KinVtxFitter {
public:
  KinVtxFitter() : fitted_vtx_{}, fitted_state_{}, fitted_particle_{}, fitted_children_{}, fitted_track_{} {};

  KinVtxFitter(const std::vector<reco::TransientTrack> tracks,
               const std::vector<double> masses,
               std::vector<float> sigmas);

  KinVtxFitter(const std::vector<reco::TransientTrack> tracks,
               const std::vector<double> masses,
               std::vector<float> sigmas,
               ParticleMass dilep_mass);

  ~KinVtxFitter() {};

  bool success() const { return success_; }
  float chi2() const { return success_ ? fitted_vtx_->chiSquared() : 999; }
  float dof() const { return success_ ? fitted_vtx_->degreesOfFreedom() : -1; }
  float prob() const { return success_ ? ChiSquaredProbability(chi2(), dof()) : 0.; }
  float kin_chi2() const { return kin_chi2_; }  // should they be merged in a single value?
  float kin_ndof() const { return kin_ndof_; }

  const KinematicState fitted_daughter(size_t i) const { return fitted_children_.at(i)->currentState(); }

  const math::PtEtaPhiMLorentzVector daughter_p4(size_t i) const {
    const auto& state = fitted_children_.at(i)->currentState();
    return math::PtEtaPhiMLorentzVector(
        state.globalMomentum().perp(), state.globalMomentum().eta(), state.globalMomentum().phi(), state.mass());
  }

  const KinematicState fitted_candidate() const { return fitted_state_; }

  const RefCountedKinematicVertex fitted_refvtx() const { return fitted_vtx_; }

  const math::PtEtaPhiMLorentzVector fitted_p4() const {
    return math::PtEtaPhiMLorentzVector(fitted_state_.globalMomentum().perp(),
                                        fitted_state_.globalMomentum().eta(),
                                        fitted_state_.globalMomentum().phi(),
                                        fitted_state_.mass());
  }

  const reco::TransientTrack& fitted_candidate_ttrk() const { return fitted_track_; }

  GlobalPoint fitted_vtx() const { return fitted_vtx_->position(); }

  GlobalError fitted_vtx_uncertainty() const { return fitted_vtx_->error(); }

private:
  float kin_chi2_ = 0.;
  float kin_ndof_ = 0.;
  size_t n_particles_ = 0;
  bool success_ = false;

  RefCountedKinematicVertex fitted_vtx_;
  KinematicState fitted_state_;
  RefCountedKinematicParticle fitted_particle_;
  std::vector<RefCountedKinematicParticle> fitted_children_;
  reco::TransientTrack fitted_track_;
};
#endif
