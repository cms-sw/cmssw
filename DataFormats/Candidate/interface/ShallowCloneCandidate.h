#ifndef Candidate_ShallowCloneCandidate_h
#define Candidate_ShallowCloneCandidate_h
/** \class reco::ShallowCloneCandidate
 *
 * shallow clone of a particle candidate keepint a reference
 * to the master clone
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ShallowCloneCandidate.h,v 1.3 2006/12/07 11:50:33 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {
  class ShallowCloneCandidate : public Candidate {
  public:
    /// collection of daughter candidates
    typedef CandidateCollection daughters;
    /// default constructor
    ShallowCloneCandidate() : Candidate() {  }
    /// constructor from Particle
    explicit ShallowCloneCandidate( const CandidateBaseRef & masterClone ) : 
      Candidate( * masterClone ), masterClone_( masterClone ) { }
    /// constructor from values
    ShallowCloneCandidate( const CandidateBaseRef & masterClone, 
			   Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      Candidate( q, p4, vtx ), masterClone_( masterClone ) { }
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
    virtual size_t numberOfDaughters() const;
    /// return daughter at a given position (throws an exception)
    virtual const Candidate * daughter( size_type i ) const;
    /// return daughter at a given position (throws an exception)
    virtual Candidate * daughter( size_type i );
    /// has master clone
    virtual bool hasMasterClone() const;
    /// returns reference to master clone
    virtual const CandidateBaseRef & masterClone() const;

    /// implementation of const_iterator. 
    /// should be private; declared public only 
    /// for ROOT reflex dictionay problems
    struct const_iterator_imp_specific : public const_iterator_imp {
      typedef ptrdiff_t difference_type;
      const_iterator_imp_specific() { }
      ~const_iterator_imp_specific() { }
      const_iterator_imp_specific * clone() const { return new const_iterator_imp_specific; }
      void increase() { }
      void decrease() { }
      void increase( difference_type d ) { }
      void decrease( difference_type d ) { }
      bool equal_to( const const_iterator_imp * o ) const { return true; }
      bool less_than( const const_iterator_imp * o ) const { return false; }
      void assign( const const_iterator_imp * o ) {  }
      const Candidate & deref() const { 
	throw cms::Exception("Invalid Dereference") << "can't dereference const_interator from LeafCandidate\n";
      }
      difference_type difference( const const_iterator_imp * o ) const { return 0; }
    };
    /// implementation of iterator. 
    /// should be private; declared public only 
    /// for ROOT reflex dictionay problems
    struct iterator_imp_specific : public iterator_imp {
      typedef ptrdiff_t difference_type;
      iterator_imp_specific() { }
      ~iterator_imp_specific() { }
      iterator_imp_specific * clone() const { return new iterator_imp_specific; }
      const_iterator_imp_specific * const_clone() const { return new const_iterator_imp_specific; }
      void increase() { }
      void decrease() { }
      void increase( difference_type d ) { }
      void decrease( difference_type d ) { }
      bool equal_to( const iterator_imp * o ) const { return true; }
      bool less_than( const iterator_imp * o ) const { return false; }
      void assign( const iterator_imp * o ) { }
      Candidate & deref() const { 
	throw cms::Exception("Invalid Dereference") << "can't dereference interator from LeafCandidate\n";
      }
      difference_type difference( const iterator_imp * o ) const { return 0; }
    };

  private:
    /// check overlap with another Candidate
    virtual bool overlap( const Candidate & c ) const { return masterClone_->overlap( c ); }
    /// CandidateBaseReference to master clone
    CandidateBaseRef masterClone_;
  };

}

#endif
