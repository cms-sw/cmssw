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
    virtual ~ShallowCloneCandidate();
    /// returns a clone of the Candidate object
    virtual ShallowCloneCandidate * clone() const;
    /// number of daughters
    virtual size_t numberOfDaughters() const;
    /// number of daughters
    virtual size_t numberOfMothers() const;
    /// return daughter at a given position (throws an exception)
    virtual const Candidate * daughter( size_type i ) const;
    /// return daughter at a given position (throws an exception)
    virtual const Candidate * mother( size_type i ) const;
    /// return daughter at a given position (throws an exception)
    virtual Candidate * daughter( size_type i );
    using reco::LeafCandidate::daughter; // avoid hiding the base
    /// has master clone
    virtual bool hasMasterClone() const;
    /// returns reference to master clone
    virtual const CandidateBaseRef & masterClone() const;

    virtual bool isElectron() const;
    virtual bool isMuon() const;
    virtual bool isGlobalMuon() const;
    virtual bool isStandAloneMuon() const;
    virtual bool isTrackerMuon() const;
    virtual bool isCaloMuon() const;
    virtual bool isPhoton() const;
    virtual bool isConvertedPhoton() const;
    virtual bool isJet() const;
  private:
    /// check overlap with another Candidate
    virtual bool overlap( const Candidate & c ) const { return masterClone_->overlap( c ); }
    /// CandidateBaseReference to master clone
    CandidateBaseRef masterClone_;
  };

}

#endif
