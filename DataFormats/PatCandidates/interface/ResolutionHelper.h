#ifndef DataFormats_PatCandidates_interface_ResolutionHelper_h
#define DataFormats_PatCandidates_interface_ResolutionHelper_h

#include "DataFormats/PatCandidates/interface/CandKinResolution.h"

namespace pat {
  namespace helper {
    namespace ResolutionHelper {
      void rescaleForKinFitter(const pat::CandKinResolution::Parametrization parametrization,
                               AlgebraicSymMatrix44 &covariance,
                               const math::XYZTLorentzVector &initialP4);
      double getResolEta(pat::CandKinResolution::Parametrization parametrization,
                         const AlgebraicSymMatrix44 &covariance,
                         const pat::CandKinResolution::LorentzVector &p4);
      double getResolTheta(pat::CandKinResolution::Parametrization parametrization,
                           const AlgebraicSymMatrix44 &covariance,
                           const pat::CandKinResolution::LorentzVector &p4);
      double getResolPhi(pat::CandKinResolution::Parametrization parametrization,
                         const AlgebraicSymMatrix44 &covariance,
                         const pat::CandKinResolution::LorentzVector &p4);
      double getResolE(pat::CandKinResolution::Parametrization parametrization,
                       const AlgebraicSymMatrix44 &covariance,
                       const pat::CandKinResolution::LorentzVector &p4);
      double getResolEt(pat::CandKinResolution::Parametrization parametrization,
                        const AlgebraicSymMatrix44 &covariance,
                        const pat::CandKinResolution::LorentzVector &p4);
      double getResolM(pat::CandKinResolution::Parametrization parametrization,
                       const AlgebraicSymMatrix44 &covariance,
                       const pat::CandKinResolution::LorentzVector &p4);
      double getResolP(pat::CandKinResolution::Parametrization parametrization,
                       const AlgebraicSymMatrix44 &covariance,
                       const pat::CandKinResolution::LorentzVector &p4);
      double getResolPt(pat::CandKinResolution::Parametrization parametrization,
                        const AlgebraicSymMatrix44 &covariance,
                        const pat::CandKinResolution::LorentzVector &p4);
      double getResolPInv(pat::CandKinResolution::Parametrization parametrization,
                          const AlgebraicSymMatrix44 &covariance,
                          const pat::CandKinResolution::LorentzVector &p4);
      double getResolPx(pat::CandKinResolution::Parametrization parametrization,
                        const AlgebraicSymMatrix44 &covariance,
                        const pat::CandKinResolution::LorentzVector &p4);
      double getResolPy(pat::CandKinResolution::Parametrization parametrization,
                        const AlgebraicSymMatrix44 &covariance,
                        const pat::CandKinResolution::LorentzVector &p4);
      double getResolPz(pat::CandKinResolution::Parametrization parametrization,
                        const AlgebraicSymMatrix44 &covariance,
                        const pat::CandKinResolution::LorentzVector &p4);
    }  // namespace ResolutionHelper
  }    // namespace helper
}  // namespace pat

#endif
