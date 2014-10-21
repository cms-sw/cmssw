#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

using namespace reco;

VertexCompositePtrCandidate::VertexCompositePtrCandidate(Charge q, const LorentzVector & p4, const Point & vtx,
						   const CovarianceMatrix & err, double chi2, double ndof,
						   int pdgId, int status, bool integerCharge) :
  CompositePtrCandidate(q, p4, vtx, pdgId, status, integerCharge),
  chi2_(chi2), ndof_(ndof) { 
  setCovariance(err);
}

VertexCompositePtrCandidate::~VertexCompositePtrCandidate() { }

VertexCompositePtrCandidate * VertexCompositePtrCandidate::clone() const { 
  return new VertexCompositePtrCandidate(*this); 
}

void VertexCompositePtrCandidate::fillVertexCovariance(CovarianceMatrix& err) const {
  index idx = 0;
  for(index i = 0; i < dimension; ++i) 
    for(index j = 0; j <= i; ++ j)
      err(i, j) = covariance_[idx++];
}

void VertexCompositePtrCandidate::setCovariance(const CovarianceMatrix & err) {
  index idx = 0;
  for(index i = 0; i < dimension; ++i) 
    for(index j = 0; j <= i; ++j)
      covariance_[idx++] = err(i, j);
}
