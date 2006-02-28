#ifndef Candidate_LeafCandidate_h
#define Candidate_LeafCandidate_h
// $Id: LeafCandidate.h,v 1.12 2006/02/21 10:37:32 llista Exp $
#include "DataFormats/Candidate/interface/Candidate.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace reco {
  
  class LeafCandidate : public Candidate {
  public:
    typedef CandidateCollection daughters;
    struct const_iterator_leaf : public const_iterator_imp {
      typedef ptrdiff_t difference_type;
      const_iterator_leaf() { }
      ~const_iterator_leaf() { }
      const_iterator_leaf * clone() const { return new const_iterator_leaf; }
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
    struct iterator_leaf : public iterator_imp {
      typedef ptrdiff_t difference_type;
      iterator_leaf() { }
      ~iterator_leaf() { }
      iterator_leaf * clone() const { return new iterator_leaf; }
      const_iterator_leaf * const_clone() const { return new const_iterator_leaf; }
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

    LeafCandidate() : Candidate() { }
    explicit LeafCandidate( const Particle & p ) : Candidate( p ) { }
    explicit LeafCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) : 
      Candidate( q, p4, vtx ) { }
    virtual ~LeafCandidate();
    virtual LeafCandidate * clone() const;

    virtual const_iterator begin() const;
    virtual const_iterator end() const;
    virtual iterator begin();
    virtual iterator end();
    virtual int numberOfDaughters() const;
    virtual const Candidate & daughter( size_type ) const;
    virtual Candidate & daughter( size_type );

  private:
    virtual bool overlap( const Candidate & c ) const;
  };

}

#endif
