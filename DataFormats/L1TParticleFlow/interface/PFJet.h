#ifndef DataFormats_L1TParticleFlow_PFJet_h
#define DataFormats_L1TParticleFlow_PFJet_h

#include <vector>
#include "DataFormats/L1Trigger/interface/L1Candidate.h"
#include "DataFormats/L1TParticleFlow/interface/PFCandidate.h"
#include "DataFormats/L1TParticleFlow/interface/jets.h"
#include "DataFormats/Common/interface/Ptr.h"

namespace l1t {

  class PFJet : public L1Candidate {
  public:
    /// constituent information. note that this is not going to be available in the hardware!
    typedef std::vector<edm::Ptr<l1t::PFCandidate>> Constituents;

    PFJet() {}
    PFJet(float pt, float eta, float phi, float mass, int hwpt = 0, int hweta = 0, int hwphi = 0)
        : L1Candidate(PolarLorentzVector(pt, eta, phi, mass), hwpt, hweta, hwphi, /*hwQuality=*/0), rawPt_(pt) {}

    PFJet(const LorentzVector& p4, int hwpt = 0, int hweta = 0, int hwphi = 0)
        : L1Candidate(p4, hwpt, hweta, hwphi, /*hwQuality=*/0), rawPt_(p4.Pt()) {}

    // change the pt (but doesn't change the raw pt)
    void calibratePt(float newpt);

    // return the raw pT()
    float rawPt() const { return rawPt_; }

    /// constituent information. note that this is not going to be available in the hardware!
    const Constituents& constituents() const { return constituents_; }
    /// adds a candidate to this cluster; note that this only records the information, it's up to you to also set the 4-vector appropriately
    void addConstituent(const edm::Ptr<l1t::PFCandidate>& cand) { constituents_.emplace_back(cand); }

    // add jet tag prediction results
    void addTagScores(std::vector<float> scores, std::vector<l1ct::JetTagClass> classes, float ptcorrection) {
      tagScores_ = scores;
      tagClasses_ = classes;
      ptCorrection_ = ptcorrection;
    }

    std::vector<float> getTagScores() const { return tagScores_; }
    std::vector<l1ct::JetTagClass> getTagClasses() const { return tagClasses_; }

    float getTagScore(l1ct::JetTagClass tagClass) const {
      // get the tag score for a specific tagClass
      auto it = std::find(tagClasses_.begin(), tagClasses_.end(), tagClass);
      if (it != tagClasses_.end()) {
        return tagScores_[std::distance(tagClasses_.begin(), it)];
      }
      return -1.0f;  // Return an invalid/default score if tagClass is not found
    }

    float getTagScore(const std::string& tagClassStr) const {
      // get the tag score for a specific tagClass
      l1ct::JetTagClass tagClass(tagClassStr);
      return getTagScore(tagClass);
    }

    float getPtCorrection() const { return ptCorrection_; }

    // candidate interface
    size_t numberOfDaughters() const override { return constituents_.size(); }
    const reco::Candidate* daughter(size_type i) const override { return constituents_[i].get(); }
    using reco::LeafCandidate::daughter;  // avoid hiding the base
    edm::Ptr<l1t::PFCandidate> daughterPtr(size_type i) const { return constituents_[i]; }

    // Get and set the encodedJet_ bits. The Jet is encoded in 128 bits as a 2-element array of uint64_t
    // We store encodings both for Correlator internal usage and for Global Trigger
    enum class HWEncoding { CT, GT, GTWide };
    typedef std::array<uint64_t, 2> PackedJet;
    const PackedJet& encodedJet(const HWEncoding encoding = HWEncoding::GT) const {
      return encodedJet_[static_cast<int>(encoding)];
    }
    void setEncodedJet(const HWEncoding encoding, const PackedJet jet) {
      encodedJet_[static_cast<int>(encoding)] = jet;
    }

    // Accessors to HW objects with ap_* types from encoded words
    const PackedJet& getHWJetGT() const { return encodedJet(HWEncoding::GT); }
    const PackedJet& getHWJetGTWide() const { return encodedJet(HWEncoding::GTWide); }
    const PackedJet& getHWJetCT() const { return encodedJet(HWEncoding::CT); }

  private:
    float rawPt_;
    Constituents constituents_;
    std::vector<l1ct::JetTagClass> tagClasses_;
    std::vector<float> tagScores_;
    float ptCorrection_;
    std::array<PackedJet, 3> encodedJet_ = {{{{0, 0}}, {{0, 0}}, {{0, 0}}}};
  };

  typedef std::vector<l1t::PFJet> PFJetCollection;
  typedef edm::Ref<l1t::PFJetCollection> PFJetRef;
  typedef edm::RefVector<l1t::PFJetCollection> PFJetRefVector;
  typedef std::vector<l1t::PFJetRef> PFJetVectorRef;
}  // namespace l1t
#endif
