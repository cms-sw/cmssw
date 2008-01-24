// $Id: VertexCompositeCandidate.cc,v 1.2 2007/11/21 12:58:42 llista Exp $
#include "DataFormats/Candidate/interface/VertexCompositeCandidate.h"

using namespace reco;

const unsigned int VertexCompositeCandidate::longLivedTag = 65536;


VertexCompositeCandidate::VertexCompositeCandidate(Charge q, const LorentzVector & p4, const Point & vtx,
						   const CovarianceMatrix & err, double chi2, double ndof,
						   int pdgId, int status, bool integerCharge) :
  CompositeCandidate(q, p4, vtx, pdgId, status, integerCharge),
  chi2_(chi2), ndof_(ndof) { 
  setCovariance(err);
}

VertexCompositeCandidate::~VertexCompositeCandidate() { }

VertexCompositeCandidate * VertexCompositeCandidate::clone() const { 
  return new VertexCompositeCandidate(*this); 
}

void VertexCompositeCandidate::fillVertexCovariance(CovarianceMatrix& err) const {
  index idx = 0;
  for(index i = 0; i < dimension; ++i) 
    for(index j = 0; j <= i; ++ j)
      err(i, j) = covariance_[idx++];
}

void VertexCompositeCandidate::setCovariance(const CovarianceMatrix & err) {
  index idx = 0;
  for(index i = 0; i < dimension; ++i) 
    for(index j = 0; j <= i; ++j)
      covariance_[idx++] = err(i, j);
}
