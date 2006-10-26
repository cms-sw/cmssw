#ifndef Candidate_CompositeCandidate_H
#define Candidate_CompositeCandidate_H
#include "DataFormats/Candidate/interface/Candidate.h"
/** \class reco::CompositeCandidate
 *
 * A Candidate composed of daughters. 
 * The daughters are owned by the composite candidate.
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: CompositeCandidate.h,v 1.4 2006/07/24 06:33:58 llista Exp $
 *
 */
namespace reco {

  class CompositeCandidate : public Candidate {
  public:
    /// collection of daughters
    typedef CandidateCollection daughters;
    /// default constructor
    CompositeCandidate() : Candidate() { }
    /// constructor from values
    CompositeCandidate( Charge q, const LorentzVector & p4, const Point & vtx = Point( 0, 0, 0 ) ) :
      Candidate( q, p4, vtx ) { }
    /// destructor
    virtual ~CompositeCandidate();
    /// returns a clone of the candidate
    virtual CompositeCandidate * clone() const;
    /// first daughter const_iterator
    virtual const_iterator begin() const;
    /// last daughter const_iterator
    virtual const_iterator end() const;
    /// first daughter iterator
    virtual iterator begin();
    /// last daughter const_iterator
    virtual iterator end();
    /// number of daughters
    virtual int numberOfDaughters() const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1 (read only mode)
    virtual const Candidate & daughter( size_type ) const;
    /// return daughter at a given position, i = 0, ... numberOfDaughters() - 1
    virtual Candidate & daughter( size_type );
    /// add a clone of the passed candidate as daughter 
    void addDaughter( const Candidate & );

    /// implementation of const_iterator. 
    /// should be private; declared public only 
    /// for ROOT reflex dictionay problems    
    struct const_iterator_comp : public const_iterator_imp {
      typedef ptrdiff_t difference_type;
      explicit const_iterator_comp( const daughters::const_iterator & it ) : i ( it ) { }
      ~const_iterator_comp() { }
      const_iterator_comp * clone() const { return new const_iterator_comp( i ); }
      void increase() { ++i; }
      void decrease() { --i; }
      void increase( difference_type d ) { i += d; }
      void decrease( difference_type d ) { i -= d; }
      bool equal_to( const const_iterator_imp * o ) const { return i == dc( o ); }
      bool less_than( const const_iterator_imp * o ) const { return i < dc( o ); }
      void assign( const const_iterator_imp * o ) { i = dc( o ); }
      const Candidate & deref() const { return * i; }
      difference_type difference( const const_iterator_imp * o ) const { return i - dc( o ); }
    private:
      const daughters::const_iterator & dc( const const_iterator_imp * o ) const {
	return dynamic_cast<const const_iterator_comp *>( o )->i;
      }
      daughters::const_iterator & dc( const_iterator_imp * o ) const {
	return dynamic_cast<const_iterator_comp *>( o )->i;
      }
      daughters::const_iterator i;
    };
    /// implementation of iterator. 
    /// should be private; declared public only 
    /// for ROOT reflex dictionay problems
    struct iterator_comp : public iterator_imp  {
      typedef ptrdiff_t difference_type;
      explicit iterator_comp( const daughters::iterator & it ) : i ( it ) { }
      ~iterator_comp() { }
      iterator_comp * clone() const { return new iterator_comp( i ); }
      const_iterator_comp * const_clone() const { return new const_iterator_comp( i ); }
      void increase() { ++i; }
      void decrease() { ++i; }
      void increase( difference_type d ) { i += d; }
      void decrease( difference_type d ) { i -= d; }
      bool equal_to( const iterator_imp * o ) const { return i == dc( o ); }
      bool less_than( const iterator_imp * o ) const { return i < dc( o ); }
      void assign( const iterator_imp * o ) { i = dc( o ); }
      Candidate & deref() const { return * i; }
      difference_type difference( const iterator_imp * o ) const { return i - dc( o ); }
     private:
      const daughters::iterator & dc( const iterator_imp * o ) const {
	return dynamic_cast<const iterator_comp *>( o )->i;
      }
      daughters::iterator & dc( iterator_imp * o ) const {
	return dynamic_cast<iterator_comp *>( o )->i;
      }
      daughters::iterator i;
    };

  private:
    /// collection of daughters
    daughters dau;
    /// check overlap with another daughter
    virtual bool overlap( const Candidate & ) const;
  };

  inline void CompositeCandidate::addDaughter( const Candidate & cand ) { 
    Candidate * c = cand.clone();
    dau.push_back( c ); 
  }
}

#endif
