#ifndef RecoCandUtils_CandCommonVertexFitter_h
#define RecoCandUtils_CandCommonVertexFitter_h
/* \class CandCommonVertexFitter
 *
 * \author Luca Lista, INFN
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/Candidate/interface/VertexCompositeCandidateOnlyFwd.h"
#include <vector>
class MagneticField;

class CandCommonVertexFitterBase {
public:
  typedef reco::Vertex::CovarianceMatrix CovarianceMatrix;
  CandCommonVertexFitterBase(const edm::ParameterSet &) : bField_(nullptr) {}
  virtual ~CandCommonVertexFitterBase() {}
  void set(const MagneticField *bField) { bField_ = bField; }
  void set(reco::VertexCompositeCandidate &) const;

protected:
  const MagneticField *bField_;
  void fill(std::vector<reco::TransientTrack> &,
            std::vector<reco::Candidate *> &,
            std::vector<reco::RecoCandidate::TrackType> &,
            reco::Candidate &) const;
  virtual bool fit(TransientVertex &, const std::vector<reco::TransientTrack> &) const = 0;
  /// chi-sqared
  mutable double chi2_;
  /// number of degrees of freedom
  mutable double ndof_;
  /// covariance matrix (3x3)
  mutable CovarianceMatrix cov_;
};

#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"

template <typename Fitter>
class CandCommonVertexFitter : public CandCommonVertexFitterBase {
public:
  CandCommonVertexFitter(const edm::ParameterSet &cfg)
      : CandCommonVertexFitterBase(cfg), fitter_(reco::modules::make<Fitter>(cfg)) {}
  bool fit(TransientVertex &vertex, const std::vector<reco::TransientTrack> &tracks) const override {
    try {
      vertex = fitter_.vertex(tracks);
    } catch (std::exception &err) {
      std::cerr << ">>> exception thrown by KalmanVertexFitter:\n"
                << err.what() << "\n"
                << ">>> candidate not fitted to common vertex" << std::endl;
      return false;
    }
    return vertex.isValid();
  }

private:
  Fitter fitter_;
};

#endif
