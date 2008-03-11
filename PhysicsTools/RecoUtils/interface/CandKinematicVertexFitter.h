#ifndef RecoCandUtils_CandKinematicVertexFitter_h
#define RecoCandUtils_CandKinematicVertexFitter_h
/* \class CandKinematicVertexFitter
 *
 * \author Luca Lista, INFN
 *
 */

#include "FWCore/ParameterSet/interface/ParameterSetfwd.h"
#include "TrackingTools/TransientTrack/interface/TransientTrack.h"
#include "RecoVertex/VertexPrimitives/interface/TransientVertex.h"
#include "DataFormats/RecoCandidate/interface/RecoCandidate.h"
#include "RecoVertex/KinematicFit/interface/KinematicParticleVertexFitter.h"
#include "PhysicsTools/UtilAlgos/interface/ParameterAdapter.h"
#include "RecoVertex/KinematicFitPrimitives/interface/KinematicParticleFactoryFromTransientTrack.h"
#include <vector>
#include "boost/shared_ptr.hpp"

class MagneticField;
namespace reco { class VertexCompositeCandidate; }

class CandKinematicVertexFitter {
public:
  typedef reco::Vertex::CovarianceMatrix CovarianceMatrix;
  CandKinematicVertexFitter(const edm::ParameterSet & cfg) :  
    bField_(0), fitter_(), fitters_(new std::vector<CandKinematicVertexFitter>) { 
  }
  CandKinematicVertexFitter(const CandKinematicVertexFitter& o) :
    bField_(o.bField_), fitter_(), fitters_(new std::vector<CandKinematicVertexFitter>) {
  }
  void set(const MagneticField * bField) { bField_ = bField; }
  void set(reco::VertexCompositeCandidate &) const;
  bool fit(const std::vector<RefCountedKinematicParticle> & tracks) const;
  RefCountedKinematicParticle currentParticle() const {
    tree_->movePointerToTheTop();
    return tree_->currentParticle();
  }
private:
  const MagneticField * bField_;
  void fill(std::vector<RefCountedKinematicParticle> &,
	    std::vector<reco::Candidate *> &,
	    std::vector<reco::RecoCandidate::TrackType> &,
	    reco::Candidate &) const;
  /// fitter
  KinematicParticleVertexFitter fitter_;
  /// fit tree
  mutable RefCountedKinematicTree tree_;
  /// particle factor
  KinematicParticleFactoryFromTransientTrack factory_;
  /// chi-sqared
  mutable double chi2_;
  /// number of degrees of freedom
  mutable double ndof_;
  /// covariance matrix (3x3)
  mutable CovarianceMatrix cov_;
  /// fitters used for recursive calls
  boost::shared_ptr<std::vector<CandKinematicVertexFitter> > fitters_;
};

#endif
