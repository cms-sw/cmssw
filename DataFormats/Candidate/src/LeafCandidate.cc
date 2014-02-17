// $Id: LeafCandidate.cc,v 1.15 2012/10/13 07:39:04 innocent Exp $
#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace reco;

LeafCandidate::~LeafCandidate() { }

Candidate::const_iterator LeafCandidate::begin() const { 
  return const_iterator( new const_iterator_imp_specific ); 
}

Candidate::const_iterator LeafCandidate::end() const { 
  return  const_iterator( new const_iterator_imp_specific ); 
}

Candidate::iterator LeafCandidate::begin() { 
  return iterator( new iterator_imp_specific ); 
}

Candidate::iterator LeafCandidate::end() { 
  return iterator( new iterator_imp_specific ); 
}

const CandidateBaseRef & LeafCandidate::masterClone() const {
  throw cms::Exception("Invalid Reference")
    << "this Candidate has no master clone reference."
    << "Can't call masterClone() method.\n";
}

bool LeafCandidate::hasMasterClone() const {
  return false;
}

bool LeafCandidate::hasMasterClonePtr() const {
  return false;
}


const CandidatePtr & LeafCandidate::masterClonePtr() const {
  throw cms::Exception("Invalid Reference")
    << "this Candidate has no master clone ptr."
    << "Can't call masterClonePtr() method.\n";
}

size_t LeafCandidate::numberOfDaughters() const { 
  return 0; 
}

size_t LeafCandidate::numberOfMothers() const { 
  return 0; 
}

bool LeafCandidate::overlap( const Candidate & o ) const { 
  return  p4() == o.p4() && vertex() == o.vertex() && charge() == o.charge();
}

const Candidate * LeafCandidate::daughter( size_type ) const {
  return 0;
}

const Candidate * LeafCandidate::mother( size_type ) const {
  return 0;
}

const Candidate * LeafCandidate::daughter(const std::string&) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "This Candidate type does not implement daughter(std::string). "
    << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
}

Candidate * LeafCandidate::daughter(const std::string&) {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "This Candidate type does not implement daughter(std::string). "
    << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
}



Candidate * LeafCandidate::daughter( size_type ) {
  return 0;
}

double LeafCandidate::vertexChi2() const {
  return 0;
}

double LeafCandidate::vertexNdof() const {
  return 0;
}

double LeafCandidate::vertexNormalizedChi2() const {
  return 0;
}

double LeafCandidate::vertexCovariance(int i, int j) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "reco::ConcreteCandidate does not implement vertex covariant matrix.\n";
}

void LeafCandidate::fillVertexCovariance(CovarianceMatrix & err) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "reco::ConcreteCandidate does not implement vertex covariant matrix.\n";
}

bool LeafCandidate::isElectron() const { return false; }

bool LeafCandidate::isMuon() const { return false; }

bool LeafCandidate::isGlobalMuon() const { return false; }

bool LeafCandidate::isStandAloneMuon() const { return false; }

bool LeafCandidate::isTrackerMuon() const { return false; }

bool LeafCandidate::isCaloMuon() const { return false; }

bool LeafCandidate::isPhoton() const { return false; }

bool LeafCandidate::isConvertedPhoton() const { return false; }

bool LeafCandidate::isJet() const { return false; }

const unsigned int reco::LeafCandidate::longLivedTag = 65536;

const unsigned int reco::LeafCandidate::massConstraintTag = 131072;

