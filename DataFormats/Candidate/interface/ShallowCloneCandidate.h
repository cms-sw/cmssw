#ifndef Candidate_ShallowCloneCandidate_h
#define Candidate_ShallowCloneCandidate_h
/** \class reco::ShallowCloneCandidate
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
  class ShallowCloneCandidate : public LeafCandidate {
  public:
    /// collection of daughter candidates
    typedef CandidateCollection daughters;
    /// default constructor
    ShallowCloneCandidate() : LeafCandidate() {  }
    /// constructor from Particle
    explicit ShallowCloneCandidate( const CandidateBaseRef & masterClone ) : 
      LeafCandidate( * masterClone ), 
      masterClone_( masterClone->hasMasterClone() ? 
		    masterClone->masterClone() : 
		    masterClone ) { 
    }
    /// constructor from values
    ShallowCloneCandidate( const CandidateBaseRef & masterClone, 
			   Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      LeafCandidate( q, p4, vtx ), masterClone_( masterClone ) { }
    /// constructor from values
    ShallowCloneCandidate( const CandidateBaseRef & masterClone, 
			   Charge q, const PolarLorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      LeafCandidate( q, p4, vtx ), masterClone_( masterClone ) { }
    /// destructor
    ~ShallowCloneCandidate() override;
    /// returns a clone of the Candidate object
    ShallowCloneCandidate * clone() const override;
    /// number of daughters
    size_t numberOfDaughters() const override;
    /// number of daughters
    size_t numberOfMothers() const override;
    /// return daughter at a given position (throws an exception)
    const Candidate * daughter( size_type i ) const override;
    /// return daughter at a given position (throws an exception)
    const Candidate * mother( size_type i ) const override;
    /// return daughter at a given position (throws an exception)
    Candidate * daughter( size_type i ) override;
    using reco::LeafCandidate::daughter; // avoid hiding the base
    /// has master clone
    bool hasMasterClone() const override;
    /// returns reference to master clone
    const CandidateBaseRef & masterClone() const override;

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
    bool overlap( const Candidate & c ) const override { return masterClone_->overlap( c ); }
    /// CandidateBaseReference to master clone
    CandidateBaseRef masterClone_;
  };

}

#endif
