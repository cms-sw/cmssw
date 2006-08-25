#ifndef Candidate_ShallowCloneCandidate_h
#define Candidate_ShallowCloneCandidate_h
/** \class reco::ShallowCloneCandidate
 *
 * shallow clone of a particle candidate keepint a reference
 * to the master clone
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ShallowCloneCandidate.h,v 1.3 2006/07/24 06:33:58 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/EDMException.h"

namespace reco {
  
  template<typename Ref>
  class ShallowCloneCandidate : public Candidate {
  public:
    /// default constructor
    ShallowCloneCandidate() : Candidate() { }
    /// constructor from Particle
    explicit ShallowCloneCandidate( const Ref & master ) : 
      Candidate( * master ), master_( master ) { }
    /// constructor from values
    ShallowCloneCandidate( const Ref & master, 
			   Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      Candidate( q, p4, vtx ), master_( master ) { }
    /// destructor
    virtual ~ShallowCloneCandidate() { }
    /// returns a clone of the Candidate object
    virtual ShallowCloneCandidate * clone() const { return new ShallowCloneCandidate( *this ); }
    /// first daughter const_iterator
    virtual const_iterator begin() const { return master_->begin(); }
    /// last daughter const_iterator
    virtual const_iterator end() const { return master_->end(); }
    /// first daughter iterator
    virtual iterator begin() { 
      throw cms::Exception("Invalid Dereference") << "can't have non-const access to master clone\n";      
    }
    /// last daughter iterator
    virtual iterator end() { 
      throw cms::Exception("Invalid Dereference") << "can't have non-const access to master clone\n";      
    }
    /// number of daughters
    virtual int numberOfDaughters() const { return master_->numberOfDaughters(); }
    /// return daughter at a given position (throws an exception)
    virtual const Candidate & daughter( size_type i ) const { return master_->daughter( i ); }
    /// return daughter at a given position (throws an exception)
    virtual Candidate & daughter( size_type i ) { 
      throw cms::Exception("Invalid Dereference") << "can't have non-const access to master clone\n";      
    }
  private:
    /// check overlap with another Candidate
    virtual bool overlap( const Candidate & c ) const { return master_->overlap( c ); }
    /// reference to master clone
    Ref master_;
  };

}

#endif
