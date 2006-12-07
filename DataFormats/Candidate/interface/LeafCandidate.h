#ifndef Candidate_LeafCandidate_h
#define Candidate_LeafCandidate_h
/** \class reco::LeafCandidate
 *
 * particle candidate with no constituent nor daughters
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: LeafCandidate.h,v 1.4 2006/12/05 15:52:59 llista Exp $
 *
 */
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace reco {
  
  class LeafCandidate : public Candidate {
  public:
    /// collection of daughter candidates
    typedef CandidateCollection daughters;
    /// default constructor
    LeafCandidate() : Candidate() { }
    /// constructor from Particle
    explicit LeafCandidate( const Particle & p ) : Candidate( p ) { }
    /// constructor from values
    LeafCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      Candidate( q, p4, vtx ) { }
    /// destructor
    virtual ~LeafCandidate();
    /// returns a clone of the Candidate object
    virtual LeafCandidate * clone() const;
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
    virtual const Candidate * daughter( size_type ) const;
    /// return daughter at a given position (throws an exception)
    virtual Candidate * daughter( size_type );
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
    virtual bool overlap( const Candidate & c ) const;
  };

}

#endif
