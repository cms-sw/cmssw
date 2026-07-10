#include "DataFormats/PatCandidates/interface/PackedGenParticle.h"
#include "DataFormats/Math/interface/libminifloat.h"
#include "DataFormats/Math/interface/deltaPhi.h"

namespace pat::io_v1 {
  void PackedGenParticle::pack(bool unpackAfterwards) {
    packedPt_ = MiniFloatConverter::float32to16(p4_.load()->Pt());
    packedY_ = int16_t(p4_.load()->Rapidity() / 6.0f * std::numeric_limits<int16_t>::max());
    packedPhi_ = int16_t(p4_.load()->Phi() / 3.2f * std::numeric_limits<int16_t>::max());
    packedM_ = MiniFloatConverter::float32to16(p4_.load()->M());
    if (unpackAfterwards) {
      delete p4_.exchange(nullptr);
      delete p4c_.exchange(nullptr);
      unpack();  // force the values to match with the packed ones
    }
  }

  void PackedGenParticle::unpack() const {
    float y = int16_t(packedY_) * 6.0f / std::numeric_limits<int16_t>::max();
    float pt = MiniFloatConverter::float16to32(packedPt_);
    float m = MiniFloatConverter::float16to32(packedM_);
    float pz = std::tanh(y) * std::sqrt((m * m + pt * pt) / (1. - std::tanh(y) * std::tanh(y)));
    float eta = 0;
    if (pt != 0.) {
      eta = std::asinh(pz / pt);
    }
    double shift = (pt < 1. ? 0.1 * pt : 0.1 / pt);  // shift particle phi to break degeneracies in angular separations
    double sign = ((int(pt * 10) % 2 == 0) ? 1 : -1);  // introduce a pseudo-random sign of the shift
    double phi = int16_t(packedPhi_) * 3.2f / std::numeric_limits<int16_t>::max() +
                 sign * shift * 3.2 / std::numeric_limits<int16_t>::max();
    auto p4 = std::make_unique<PolarLorentzVector>(pt, eta, phi, m);
    PolarLorentzVector* expectp4 = nullptr;
    if (p4_.compare_exchange_strong(expectp4, p4.get())) {
      p4.release();
    }
    auto p4c = std::make_unique<LorentzVector>(*p4_);
    LorentzVector* expectp4c = nullptr;
    if (p4c_.compare_exchange_strong(expectp4c, p4c.get())) {
      p4c.release();
    }
  }

  PackedGenParticle::~PackedGenParticle() {
    delete p4_.load();
    delete p4c_.load();
  }

  float PackedGenParticle::dxy(const Point& p) const {
    unpack();
    return -(vertex_.X() - p.X()) * std::sin(float(p4_.load()->Phi())) +
           (vertex_.Y() - p.Y()) * std::cos(float(p4_.load()->Phi()));
  }
  float PackedGenParticle::dz(const Point& p) const {
    unpack();
    return (vertex_.Z() - p.X()) - ((vertex_.X() - p.X()) * std::cos(float(p4_.load()->Phi())) +
                                    (vertex_.Y() - p.Y()) * std::sin(float(p4_.load()->Phi()))) *
                                       p4_.load()->Pz() / p4_.load()->Pt();
  }

  //// Everything below is just trivial implementations of reco::Candidate methods

  const reco::CandidateBaseRef& PackedGenParticle::masterClone() const {
    throw cms::Exception("Invalid Reference") << "this Candidate has no master clone reference."
                                              << "Can't call masterClone() method.\n";
  }

  bool PackedGenParticle::hasMasterClone() const { return false; }

  bool PackedGenParticle::hasMasterClonePtr() const { return false; }

  const reco::CandidatePtr& PackedGenParticle::masterClonePtr() const {
    throw cms::Exception("Invalid Reference") << "this Candidate has no master clone ptr."
                                              << "Can't call masterClonePtr() method.\n";
  }

  size_t PackedGenParticle::numberOfDaughters() const { return 0; }

  size_t PackedGenParticle::numberOfMothers() const {
    if (motherRef().isNonnull())
      return 1;
    return 0;
  }

  bool PackedGenParticle::overlap(const reco::Candidate& o) const {
    return p4() == o.p4() && vertex() == o.vertex() && charge() == o.charge();
    //  return  p4() == o.p4() && charge() == o.charge();
  }

  const reco::Candidate* PackedGenParticle::daughter(size_type) const { return nullptr; }

  const reco::Candidate* PackedGenParticle::mother(size_type) const { return motherRef().get(); }

  const reco::Candidate* PackedGenParticle::daughter(const std::string&) const {
    throw edm::Exception(edm::errors::UnimplementedFeature)
        << "This Candidate type does not implement daughter(std::string). "
        << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
  }

  reco::Candidate* PackedGenParticle::daughter(const std::string&) {
    throw edm::Exception(edm::errors::UnimplementedFeature)
        << "This Candidate type does not implement daughter(std::string). "
        << "Please use CompositeCandidate or NamedCompositeCandidate.\n";
  }

  reco::Candidate* PackedGenParticle::daughter(size_type) { return nullptr; }

  double PackedGenParticle::vertexChi2() const { return 0; }

  double PackedGenParticle::vertexNdof() const { return 0; }

  double PackedGenParticle::vertexNormalizedChi2() const { return 0; }

  double PackedGenParticle::vertexCovariance(int i, int j) const {
    throw edm::Exception(edm::errors::UnimplementedFeature)
        << "reco::ConcreteCandidate does not implement vertex covariant matrix.\n";
  }

  void PackedGenParticle::fillVertexCovariance(CovarianceMatrix& err) const {
    throw edm::Exception(edm::errors::UnimplementedFeature)
        << "reco::ConcreteCandidate does not implement vertex covariant matrix.\n";
  }

  bool PackedGenParticle::isElectron() const { return false; }

  bool PackedGenParticle::isMuon() const { return false; }

  bool PackedGenParticle::isGlobalMuon() const { return false; }

  bool PackedGenParticle::isStandAloneMuon() const { return false; }

  bool PackedGenParticle::isTrackerMuon() const { return false; }

  bool PackedGenParticle::isCaloMuon() const { return false; }

  bool PackedGenParticle::isPhoton() const { return false; }

  bool PackedGenParticle::isConvertedPhoton() const { return false; }

  bool PackedGenParticle::isJet() const { return false; }

  bool PackedGenParticle::longLived() const { return false; }

  bool PackedGenParticle::massConstraint() const { return false; }
}  // namespace pat::io_v1
