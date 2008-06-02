#ifndef Candidate_iterator_h
#define Candidate_iterator_h

/* \class reco::candidate::iterator
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/iterator_imp.h"
#include "DataFormats/Candidate/interface/const_iterator.h"

namespace reco {
  namespace candidate {
    struct iterator {
      typedef Candidate value_type;
      typedef Candidate * pointer;
      typedef Candidate & reference;
      typedef ptrdiff_t difference_type;
      typedef std::vector<int>::iterator::iterator_category iterator_category;
      iterator() : i( 0 ) { }
      iterator( iterator_imp * it ) : i( it ) { }
      iterator( const iterator & it ) : i( it.i->clone() ) { }
      ~iterator() { delete i; }
      iterator & operator=( const iterator & it ) { i->assign( it.i ); return *this; }
      iterator& operator++() { i->increase(); return *this; }
      iterator operator++( int ) { iterator ci = *this; i->increase(); return ci; }
      iterator& operator--() { i->increase(); return *this; }
      iterator operator--( int ) { iterator ci = *this; i->decrease(); return ci; }
      difference_type operator-( const iterator & o ) const { return i->difference( o.i ); }
      iterator operator+( difference_type n ) const { 
	iterator_imp * ii = i->clone(); ii->increase( n );
	return iterator( ii ); 
      }
      iterator operator-( difference_type n ) const { 
	iterator_imp * ii = i->clone(); ii->decrease( n );
	return iterator( ii ); 
      }
      bool operator<( const iterator & o ) { return i->less_than( o.i ) ; }
      bool operator==( const iterator& ci ) const { return i->equal_to( ci.i ); }
      bool operator!=( const iterator& ci ) const { return ! i->equal_to( ci.i ); }
      Candidate & operator * () const { return i->deref(); }
      Candidate * operator->() const { return & ( operator*() ); }
      iterator & operator +=( difference_type d ) { i->increase( d ); return *this; }
      iterator & operator -=( difference_type d ) { i->decrease( d ); return *this; }
    private:
      iterator_imp * i;
      friend const_iterator::const_iterator( const iterator & );
    };


  }
}

#endif
