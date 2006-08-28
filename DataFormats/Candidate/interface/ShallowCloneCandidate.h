#ifndef Candidate_ShallowCloneCandidate_h
#define Candidate_ShallowCloneCandidate_h
/** \class reco::ShallowCloneCandidate
 *
 * shallow clone of a particle candidate keepint a reference
 * to the master clone
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ShallowCloneCandidate.h,v 1.1 2006/08/25 14:36:12 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {
  class ShallowCloneCandidate : public Candidate {
  public:
    /// default constructor
    ShallowCloneCandidate() : Candidate() { hasMasterClone_ = true; }
    /// constructor from Particle
    explicit ShallowCloneCandidate( const CandidateBaseRef & masterClone ) : 
      Candidate( * masterClone ), masterClone_( masterClone ) { hasMasterClone_ = true; }
    /// constructor from values
    ShallowCloneCandidate( const CandidateBaseRef & masterClone, 
			   Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      Candidate( q, p4, vtx ), masterClone_( masterClone ) { hasMasterClone_ = true; }
    /// destructor
    virtual ~ShallowCloneCandidate();
    /// returns a clone of the Candidate object
    virtual ShallowCloneCandidate * clone() const;
    /// first daughter const_iterator
    virtual const_iterator begin() const;
    /// last daughter const_iterator
    virtual const_iterator end() const;
    /// first daughter iterator
    virtual iterator begin();
    /// last daughter iterator
    virtual iterator end();
    /// number of daughters
    virtual int numberOfDaughters() const;
    /// return daughter at a given position (throws an exception)
    virtual const Candidate & daughter( size_type i ) const;
    /// return daughter at a given position (throws an exception)
    virtual Candidate & daughter( size_type i );
    /// returns reference to master clone
    virtual const CandidateBaseRef & masterClone() const;

  private:
    /// check overlap with another Candidate
    virtual bool overlap( const Candidate & c ) const { return masterClone_->overlap( c ); }
    /// CandidateBaseReference to master clone
    CandidateBaseRef masterClone_;
  };

}

#endif
