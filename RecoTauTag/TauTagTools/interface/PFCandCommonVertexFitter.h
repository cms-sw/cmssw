#ifndef TauTagTools_PFCandCommonVertexFitter_h
#define TauTagTools_PFCandCommonVertexFitter_h

/* \class PFCandCommonVertexFitter
 *
 * Adapted from PhysicsTools/RecoUtils for use with ShallowClones of PFCandidates, which
 * have a different method for retrieving the associated transient track
 *
 * \author Luca Lista, INFN
 * Modified by Evan Friis, UC Davis
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "DataFormats/ParticleFlowCandidate/interface/PFCandidate.h"
#include <vector>
class MagneticField;
namespace reco { class VertexCompositeCandidate; }

class PFCandCommonVertexFitterBase {
public:
  typedef reco::Vertex::CovarianceMatrix CovarianceMatrix;
  PFCandCommonVertexFitterBase(const edm::ParameterSet &) : bField_(nullptr) { }
  virtual ~PFCandCommonVertexFitterBase() { }
  void set(const MagneticField * bField) { bField_ = bField; }
  void set(reco::VertexCompositeCandidate &) const;
  
protected:
  const MagneticField * bField_;
  void fill(std::vector<reco::TransientTrack> &, 
	    std::vector<reco::Candidate *> &,
	    std::vector<reco::RecoCandidate::TrackType> &,
	    reco::Candidate &) const;
  virtual bool fit(TransientVertex &, 
		   const std::vector<reco::TransientTrack> &) const = 0;
  /// chi-sqared
  mutable double chi2_;
  /// number of degrees of freedom
  mutable double ndof_;
  /// covariance matrix (3x3)
  mutable CovarianceMatrix cov_;
};

#include "CommonTools/UtilAlgos/interface/ParameterAdapter.h"

template<typename Fitter>
class PFCandCommonVertexFitter : public PFCandCommonVertexFitterBase {
public:
  PFCandCommonVertexFitter(const edm::ParameterSet & cfg) : 
    PFCandCommonVertexFitterBase(cfg),
//    fitter_(reco::modules::make<Fitter>(cfg)) { 
    fitter_(Fitter(cfg, true)) { 
  }
  bool fit(TransientVertex & vertex, 
	   const std::vector<reco::TransientTrack> & tracks) const override {
    try {
      vertex = fitter_.vertex(tracks);
    } catch (std::exception & err) {
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
