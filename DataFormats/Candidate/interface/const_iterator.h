#ifndef Candidate_const_iterator_h
#define Candidate_const_iterator_h
/* \class reco::candidate::const_iterator
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/const_iterator_imp.h"

namespace reco {
  namespace candidate {
    struct iterator;

    struct const_iterator {
      typedef const Candidate value_type;
      typedef const Candidate * pointer;
      typedef const Candidate & reference;
      typedef ptrdiff_t difference_type;
      typedef std::vector<int>::const_iterator::iterator_category iterator_category;
      const_iterator() : i( 0 ) { }
      const_iterator( const_iterator_imp * it ) : i( it ) { }
      const_iterator( const const_iterator & it ) : i( it.i->clone() ) { }
      const_iterator( const iterator & it );
      ~const_iterator() { delete i; }
      const_iterator & operator=( const const_iterator & it ) { i->assign( it.i ); return *this; }
      const_iterator& operator++() { i->increase(); return *this; }
      const_iterator operator++( int ) { const_iterator ci = *this; i->increase(); return ci; }
      const_iterator& operator--() { i->decrease(); return *this; }
      const_iterator operator--( int ) { const_iterator ci = *this; i->decrease(); return ci; }
      difference_type operator-( const const_iterator & o ) const { return i->difference( o.i ); }
      const_iterator operator+( difference_type n ) const { 
	const_iterator_imp * ii = i->clone(); ii->increase( n );
	return const_iterator( ii ); 
      }
      const_iterator operator-( difference_type n ) const { 
	const_iterator_imp * ii = i->clone(); ii->decrease( n );
	return const_iterator( ii ); 
      }
      bool operator<( const const_iterator & o ) const { return i->less_than( o.i ); }
      bool operator==( const const_iterator& ci ) const { return i->equal_to( ci.i ); }
      bool operator!=( const const_iterator& ci ) const { return ! i->equal_to( ci.i ); }
      const Candidate & operator * () const { return i->deref(); }
      const Candidate * operator->() const { return & ( operator*() ); }
      const_iterator & operator +=( difference_type d ) { i->increase( d ); return *this; }
      const_iterator & operator -=( difference_type d ) { i->decrease( d ); return *this; }

    private:
      const_iterator_imp * i;
    };
  }
}

#endif
