#ifndef Candidate_CompositeRefCandidate_h
#define Candidate_CompositeRefCandidate_h
#include "DataFormats/Candidate/interface/Candidate.h"

namespace reco {

  class CompositeRefCandidate : public Candidate {
  public:
    typedef CandidateRefs daughters;
    struct const_iterator_comp : public const_iterator_imp {
      typedef ptrdiff_t difference_type;
      const_iterator_comp() { }
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
      const Candidate & deref() const { return * * i; }
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
    struct iterator_comp : public iterator_imp {
      typedef ptrdiff_t difference_type;
      explicit iterator_comp( const daughters::iterator & it ) : i ( it ) { }
      ~iterator_comp() { }
      iterator_comp * clone() const { return new iterator_comp( i ); }
      const_iterator_comp * const_clone() const { return new const_iterator_comp( i ); }
      void increase() { ++i; }
      void decrease() { --i; }
      void increase( difference_type d ) { i += d; }
      void decrease( difference_type d ) { i -= d; }
      bool equal_to( const iterator_imp * o ) const { return i == dc( o ); }
      bool less_than( const iterator_imp * o ) const { return i < dc( o ); }
      void assign( const iterator_imp * o ) { i = dc( o ); }
      // ... nasty!
      Candidate & deref() const { return const_cast<Candidate &>( * * i ); }
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

    CompositeRefCandidate() : Candidate() { }
    virtual ~CompositeRefCandidate();
    virtual CompositeRefCandidate * clone() const;

    virtual const_iterator begin() const;
    virtual const_iterator end() const;
    virtual iterator begin();
    virtual iterator end();
    virtual int numberOfDaughters() const;
    virtual const Candidate & daughter( size_type ) const;
    virtual Candidate & daughter( size_type );

    void addDaughter( const CandidateRef & );
 
  private:
    daughters dau;
    virtual bool overlap( const Candidate & ) const;
  };

  inline void CompositeRefCandidate::addDaughter( const CandidateRef & cand ) { 
    dau.push_back( cand ); 
  }
}

#endif
