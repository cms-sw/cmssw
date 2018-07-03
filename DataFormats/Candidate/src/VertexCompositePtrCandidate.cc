#include "DataFormats/Candidate/interface/VertexCompositePtrCandidate.h"

using namespace reco;

VertexCompositePtrCandidate::VertexCompositePtrCandidate(Charge q, const LorentzVector & p4, const Point & vtx,
						   const CovarianceMatrix & err, double chi2, double ndof,
						   int pdgId, int status, bool integerCharge) :
  CompositePtrCandidate(q, p4, vtx, pdgId, status, integerCharge),
  chi2_(chi2), ndof_(ndof), time_(0.) { 
  setCovariance(err);
}

VertexCompositePtrCandidate::VertexCompositePtrCandidate(Charge q, const LorentzVector & p4, const Point & vtx,
						   double time, const CovarianceMatrix4D & err, double chi2, 
						   double ndof, int pdgId, int status, bool integerCharge) :
  CompositePtrCandidate(q, p4, vtx, pdgId, status, integerCharge),
  chi2_(chi2), ndof_(ndof), time_(time) { 
  setCovariance(err);
}

VertexCompositePtrCandidate::~VertexCompositePtrCandidate() { }

VertexCompositePtrCandidate * VertexCompositePtrCandidate::clone() const { 
  return new VertexCompositePtrCandidate(*this); 
}

void VertexCompositePtrCandidate::fillVertexCovariance(CovarianceMatrix& err) const {
  Error4D temp;
  fillVertexCovariance(temp);
  err = temp.Sub<Error>(0,0);  
}

void VertexCompositePtrCandidate::fillVertexCovariance(CovarianceMatrix4D& err) const {
  index idx = 0;
  for(index i = 0; i < dimension4D; ++i) 
    for(index j = 0; j <= i; ++ j) 
      err(i, j) = covariance_[idx++];
}

void VertexCompositePtrCandidate::setCovariance(const CovarianceMatrix & err) {
  index idx = 0;
  for(index i = 0; i < dimension4D; ++i) { 
    for(index j = 0; j <= i; ++j) {
      if( i == dimension || j == dimension ) {
        covariance_[ idx ++ ] = 0.0;
      } else {
        covariance_[ idx ++ ] = err( i, j );
      }
    }
  }
}

void VertexCompositePtrCandidate::setCovariance(const CovarianceMatrix4D & err) {
  index idx = 0;
  for(index i = 0; i < dimension4D; ++i) 
    for(index j = 0; j <= i; ++j)
      covariance_[idx++] = err(i, j);
}

