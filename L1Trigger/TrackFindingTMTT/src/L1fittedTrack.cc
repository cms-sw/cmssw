#include "L1Trigger/TrackFindingTMTT/interface/L1fittedTrack.h"

using namespace std;

namespace tmtt {

  // Empty vector used to initialize rejected tracks.
  const std::vector<Stub*> L1fittedTrack::noStubs_;

  // Digitize track and degrade helix parameter resolution according to effect of digitisation.

  void L1fittedTrack::digitizeTrack(const string& fitterName) {
    if (settings_->enableDigitize()) {
      if (not digitalTrack_) {
        // Create & run digitizer.
        digitalTrack_ = std::make_shared<DigitalTrack>(settings_, fitterName, this);

        // Convert digitized track params back to floating point with degraded resolution.
        qOverPt_ = digitalTrack_->qOverPt();
        if (nHelixParam_ == 5)
          d0_ = digitalTrack_->d0();
        phi0_ = digitalTrack_->phi0();
        z0_ = digitalTrack_->z0();
        tanLambda_ = digitalTrack_->tanLambda();
        chi2rphi_ = digitalTrack_->chisquaredRphi();
        chi2rz_ = digitalTrack_->chisquaredRz();

        // Ditto for beam-spot constrained values.
        if (nHelixParam_ == 5) {
          qOverPt_bcon_ = digitalTrack_->qOverPt_bcon();
          phi0_bcon_ = digitalTrack_->phi0_bcon();
          chi2rphi_bcon_ = digitalTrack_->chisquaredRphi_bcon();
        }

        // Recalculate consistency flag using updated helix params.
        this->setConsistentHTcell();
      }
    }
  }

}  // namespace tmtt
