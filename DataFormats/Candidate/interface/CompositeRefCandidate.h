#ifndef Candidate_CompositeRefCandidate_h
#define Candidate_CompositeRefCandidate_h
#include "DataFormats/Candidate/interface/Candidate.h"
/** \class reco::CompositeRefCandidate
 *
 * a reco::Candidate composed of daughters. 
 * The daughters has persistent references (edm::Ref <...>) 
 * to reco::Candidate stored in a separate collection.
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: CompositeRefCandidate.h,v 1.5 2006/12/05 15:52:59 llista Exp $
 *
 */

namespace reco {

  class CompositeRefCandidate : public Candidate {
  public:
    /// collection of references to daughters
    typedef CandidateRefVector daughters;
    /// default constructor
    CompositeRefCandidate() : Candidate() { }
    /// constructor from values
    CompositeRefCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) :
      Candidate( q, p4, vtx ) { }
    /// destructor
    virtual ~CompositeRefCandidate();
    /// returns a clone of the candidate
    virtual CompositeRefCandidate * clone() const;
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
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    virtual const Candidate * daughter( size_type ) const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    virtual Candidate * daughter( size_type );
    /// add a daughter via a reference
    void addDaughter( const CandidateRef & );    
    /// implementation of const_iterator. 
    /// should be private; declared public only 
    /// for ROOT reflex dictionay problems    
    struct const_iterator_imp_specific : public const_iterator_imp {
      typedef ptrdiff_t difference_type;
      const_iterator_imp_specific() { }
      explicit const_iterator_imp_specific( const daughters::const_iterator & it ) : i ( it ) { }
      ~const_iterator_imp_specific() { }
      const_iterator_imp_specific * clone() const { return new const_iterator_imp_specific( i ); }
      void increase() { ++i; }
      void decrease() { --i; }
      void increase( difference_type d ) { i += d; }
      void decrease( difference_type d ) { i -= d; }
      bool equal_to( const const_iterator_imp * o ) const { return i == dc( o ); }
      bool less_than( const const_iterator_imp * o ) const { return i < dc( o ); }
      void assign( const const_iterator_imp * o ) { i = dc( o ); }
      const Candidate & deref() const { return * * i; }
      difference_type difference( const const_iterator_imp * o ) const { return i - dc( o ); }
    private:
      const daughters::const_iterator & dc( const const_iterator_imp * o ) const {
	return dynamic_cast<const const_iterator_imp_specific *>( o )->i;
      }
      daughters::const_iterator & dc( const_iterator_imp * o ) const {
	return dynamic_cast<const_iterator_imp_specific *>( o )->i;
      }
      daughters::const_iterator i;
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
    /// collection of references to daughters
    daughters dau;
    /// check overlap with another candidate
    virtual bool overlap( const Candidate & ) const;
  };

  inline void CompositeRefCandidate::addDaughter( const CandidateRef & cand ) { 
    dau.push_back( cand ); 
  }
}

#endif
