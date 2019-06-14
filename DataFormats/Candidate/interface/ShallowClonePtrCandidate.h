#ifndef Candidate_ShallowClonePtrCandidate_h
#define Candidate_ShallowClonePtrCandidate_h
/** \class reco::ShallowClonePtrCandidate
 *
 * shallow clone of a particle candidate keepint a reference
 * to the master clone
 *
 * \author Luca Lista, INFN
 *
 *
 */
#include "DataFormats/Candidate/interface/LeafCandidate.h"

namespace reco {
  class ShallowClonePtrCandidate : public LeafCandidate {
  public:
    /// collection of daughter candidates
    typedef CandidateCollection daughters;
    /// default constructor
    ShallowClonePtrCandidate() : LeafCandidate() {}
    /// constructor from Particle
    explicit ShallowClonePtrCandidate(const CandidatePtr& masterClone)
        : LeafCandidate(*masterClone), masterClone_(masterClone) {}
    /// constructor from values
    ShallowClonePtrCandidate(const CandidatePtr& masterClone,
                             Charge q,
                             const LorentzVector& p4,
                             const Point& vtx = Point(0, 0, 0))
        : LeafCandidate(q, p4, vtx), masterClone_(masterClone) {}
    /// constructor from values
    ShallowClonePtrCandidate(const CandidatePtr& masterClone,
                             Charge q,
                             const PolarLorentzVector& p4,
                             const Point& vtx = Point(0, 0, 0))
        : LeafCandidate(q, p4, vtx), masterClone_(masterClone) {}
    /// destructor
    ~ShallowClonePtrCandidate() override;
    /// returns a clone of the Candidate object
    ShallowClonePtrCandidate* clone() const override;
    /// number of daughters
    size_t numberOfDaughters() const override;
    /// number of mothers
    size_t numberOfMothers() const override;
    /// return daughter at a given position (throws an exception)
    const Candidate* daughter(size_type i) const override;
    /// return mother at a given position (throws an exception)
    const Candidate* mother(size_type i) const override;
    /// return daughter at a given position (throws an exception)
    Candidate* daughter(size_type i) override;
    using reco::LeafCandidate::daughter;  // avoid hiding the base
    /// has master clone pointer
    bool hasMasterClonePtr() const override;
    /// returns reference to master clone pointer
    const CandidatePtr& masterClonePtr() const override;

    bool isElectron() const override;
    bool isMuon() const override;
    bool isGlobalMuon() const override;
    bool isStandAloneMuon() const override;
    bool isTrackerMuon() const override;
    bool isCaloMuon() const override;
    bool isPhoton() const override;
    bool isConvertedPhoton() const override;
    bool isJet() const override;

  private:
    /// check overlap with another Candidate
    bool overlap(const Candidate& c) const override { return masterClone_->overlap(c); }
    /// CandidatePtrerence to master clone
    CandidatePtr masterClone_;
  };

}  // namespace reco

#endif
