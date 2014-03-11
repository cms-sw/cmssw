#include "DataFormats/PatCandidates/interface/PackedCandidate.h"
#include "DataFormats/PatCandidates/interface/libminifloat.h"

void pat::PackedCandidate::pack() {
    packedPt_  =  MiniFloatConverter::float32to16(p4_.Pt());
    packedEta_ =  int16_t(p4_.Eta()/6.0f*std::numeric_limits<int16_t>::max());
    packedPhi_ =  int16_t(p4_.Phi()/3.2f*std::numeric_limits<int16_t>::max());
    packedM_   =  MiniFloatConverter::float32to16(p4_.M());
    packedX_   =  int16_t(vertex_.X()/5.0f*std::numeric_limits<int16_t>::max());
    packedY_   =  int16_t(vertex_.Y()/5.0f*std::numeric_limits<int16_t>::max());
    packedZ_   =  int16_t(vertex_.Z()/40.f*std::numeric_limits<int16_t>::max());
    unpack(); // force the values to match with the packed ones
}

void pat::PackedCandidate::unpack() const {
    p4_ = PolarLorentzVector(MiniFloatConverter::float16to32(packedPt_),
                             int16_t(packedEta_)*6.0f/std::numeric_limits<int16_t>::max(),
                             int16_t(packedPhi_)*3.2f/std::numeric_limits<int16_t>::max(),
                             MiniFloatConverter::float16to32(packedM_));
    p4c_ = p4_;
    vertex_ = Point(int16_t(packedX_)*5.0f/std::numeric_limits<int16_t>::max(),
                    int16_t(packedY_)*5.0f/std::numeric_limits<int16_t>::max(),
                    int16_t(packedZ_)*40.f/std::numeric_limits<int16_t>::max());
    unpacked_ = true;
}

pat::PackedCandidate::~PackedCandidate() { }

pat::PackedCandidate::const_iterator pat::PackedCandidate::begin() const { 
  return const_iterator( new const_iterator_imp_specific ); 
}

pat::PackedCandidate::const_iterator pat::PackedCandidate::end() const { 
  return  const_iterator( new const_iterator_imp_specific ); 
}

pat::PackedCandidate::iterator pat::PackedCandidate::begin() { 
  return iterator( new iterator_imp_specific ); 
}

pat::PackedCandidate::iterator pat::PackedCandidate::end() { 
  return iterator( new iterator_imp_specific ); 
}

const reco::CandidateBaseRef & pat::PackedCandidate::masterClone() const {
  throw cms::Exception("Invalid Reference")
    << "this Candidate has no master clone reference."
    << "Can't call masterClone() method.\n";
}

bool pat::PackedCandidate::hasMasterClone() const {
  return false;
}

bool pat::PackedCandidate::hasMasterClonePtr() const {
  return false;
}


const reco::CandidatePtr & pat::PackedCandidate::masterClonePtr() const {
  throw cms::Exception("Invalid Reference")
    << "this Candidate has no master clone ptr."
    << "Can't call masterClonePtr() method.\n";
}

size_t pat::PackedCandidate::numberOfDaughters() const { 
  return 0; 
}

size_t pat::PackedCandidate::numberOfMothers() const { 
  return 0; 
}

bool pat::PackedCandidate::overlap( const reco::Candidate & o ) const { 
  return  p4() == o.p4() && vertex() == o.vertex() && charge() == o.charge();
//  return  p4() == o.p4() && charge() == o.charge();
}

const reco::Candidate * pat::PackedCandidate::daughter( size_type ) const {
  return 0;
}

const reco::Candidate * pat::PackedCandidate::mother( size_type ) const {
  return 0;
}

const reco::Candidate * pat::PackedCandidate::daughter(const std::string&) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "This Candidate type does not implement daughter(std::string). "
    << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
}

reco::Candidate * pat::PackedCandidate::daughter(const std::string&) {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "This Candidate type does not implement daughter(std::string). "
    << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
}



reco::Candidate * pat::PackedCandidate::daughter( size_type ) {
  return 0;
}

double pat::PackedCandidate::vertexChi2() const {
  return 0;
}

double pat::PackedCandidate::vertexNdof() const {
  return 0;
}

double pat::PackedCandidate::vertexNormalizedChi2() const {
  return 0;
}

double pat::PackedCandidate::vertexCovariance(int i, int j) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "reco::ConcreteCandidate does not implement vertex covariant matrix.\n";
}

void pat::PackedCandidate::fillVertexCovariance(CovarianceMatrix & err) const {
  throw edm::Exception(edm::errors::UnimplementedFeature)
    << "reco::ConcreteCandidate does not implement vertex covariant matrix.\n";
}

bool pat::PackedCandidate::isElectron() const { return false; }

bool pat::PackedCandidate::isMuon() const { return false; }

bool pat::PackedCandidate::isGlobalMuon() const { return false; }

bool pat::PackedCandidate::isStandAloneMuon() const { return false; }

bool pat::PackedCandidate::isTrackerMuon() const { return false; }

bool pat::PackedCandidate::isCaloMuon() const { return false; }

bool pat::PackedCandidate::isPhoton() const { return false; }

bool pat::PackedCandidate::isConvertedPhoton() const { return false; }

bool pat::PackedCandidate::isJet() const { return false; }

bool pat::PackedCandidate::longLived() const {return false;}

bool pat::PackedCandidate::massConstraint() const {return false;}




