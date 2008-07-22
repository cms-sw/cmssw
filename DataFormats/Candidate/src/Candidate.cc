// $Id: Candidate.cc,v 1.15 2008/05/08 15:29:35 srappocc Exp $
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"
using namespace reco;

Candidate::~Candidate() { }

const CandidateBaseRef & Candidate::masterClone() const {
  throw cms::Exception("Invalid Reference") 
    << "this Candidate has no master clone reference."
    << "Can't call masterClone() method.\n";
}

bool Candidate::hasMasterClone() const { 
  return false;
}

const CandidatePtr & Candidate::masterClonePtr() const {
  throw cms::Exception("Invalid Reference") 
    << "this Candidate has no master clone ptr."
    << "Can't call masterClonePtr() method.\n";
}

bool Candidate::hasMasterClonePtr() const { 
  return false;
}

double Candidate::vertexChi2() const { 
  return 0; 
}

double Candidate::vertexNdof() const { 
  return 0; 
}

double Candidate::vertexNormalizedChi2() const {
  return 0;
}

double Candidate::vertexCovariance(int i, int j) const { 
  throw edm::Exception(edm::errors::UnimplementedFeature) 
    << "reco::Candidate does not implement vertex covariant matrix.\n";
}

void Candidate::fillVertexCovariance(CovarianceMatrix & err) const {
  throw edm::Exception(edm::errors::UnimplementedFeature) 
    << "reco::Candidate does not implement vertex covariant matrix.\n";
}

bool Candidate::isElectron() const { return false; }
 
bool Candidate::isMuon() const { return false; }

bool Candidate::isGlobalMuon() const { return false; }

bool Candidate::isStandAloneMuon() const { return false; }

bool Candidate::isTrackerMuon() const { return false; }

bool Candidate::isCaloMuon() const { return false; }

bool Candidate::isPhoton() const { return false; }

bool Candidate::isConvertedPhoton() const { return false; }

bool Candidate::isJet() const { return false; }

const Candidate * Candidate::daughter(const std::string&) const {
  throw edm::Exception(edm::errors::UnimplementedFeature) 
    << "This Candidate type does not implement daughter(std::string). "
    << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
}

Candidate * Candidate::daughter(const std::string&) {
  throw edm::Exception(edm::errors::UnimplementedFeature) 
    << "This Candidate type does not implement daughter(std::string). "
    << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
}
