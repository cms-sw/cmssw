#ifndef L1Candidate_h
#define L1Candidate_h

#include "DataFormats/Candidate/interface/LeafCandidate.h"
#include "DataFormats/L1Trigger/interface/BXVector.h"
namespace l1t {

  class L1Candidate;
  typedef BXVector<L1Candidate> L1CandidateBxCollection;
  typedef edm::Ref<L1CandidateBxCollection> L1CandidateRef;
  typedef edm::RefVector<L1CandidateBxCollection> L1CandidateRefVector;
  typedef std::vector<L1CandidateRef> L1CandidateVectorRef;

  // All L1 data formats which encode physically meaningful quantities inherit from Candidate
  class L1Candidate : public reco::LeafCandidate {
  public:
    L1Candidate();

    // construct from *both* physical and integer values
    L1Candidate(const LorentzVector& p4, int pt = 0, int eta = 0, int phi = 0, int qual = 0, int iso = 0);

    L1Candidate(const PolarLorentzVector& p4, int pt = 0, int eta = 0, int phi = 0, int qual = 0, int iso = 0);

    ~L1Candidate() override;

    // methods to set integer values
    // in general, these should not be needed
    void setHwPt(int pt) { hwPt_ = pt; }
    void setHwEta(int eta) { hwEta_ = eta; }
    void setHwPhi(int phi) { hwPhi_ = phi; }
    void setHwQual(int qual) { hwQual_ = qual; }
    void setHwIso(int iso) { hwIso_ = iso; }

    // methods to retrieve integer values
    int hwPt() const { return hwPt_; }
    int hwEta() const { return hwEta_; }
    int hwPhi() const { return hwPhi_; }
    int hwQual() const { return hwQual_; }
    int hwIso() const { return hwIso_; }

    virtual bool operator==(const l1t::L1Candidate& rhs) const;
    virtual inline bool operator!=(const l1t::L1Candidate& rhs) const { return !(operator==(rhs)); };

  private:
    // integer "hardware" values
    int hwPt_;
    int hwEta_;
    int hwPhi_;
    int hwQual_;
    int hwIso_;
  };

};  // namespace l1t

#endif
