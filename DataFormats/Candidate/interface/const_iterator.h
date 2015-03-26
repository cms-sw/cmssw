#ifndef Candidate_const_iterator_h
#define Candidate_const_iterator_h

/* \class reco::candidate::const_iterator
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/CandidateFwd.h"
#include "DataFormats/Candidate/interface/iterator.h"

namespace reco {
  namespace candidate {
    struct const_iterator {
      typedef const Candidate value_type;
      typedef const Candidate * pointer;
      typedef const Candidate & reference;
      typedef ptrdiff_t difference_type;
      typedef std::vector<int>::const_iterator::iterator_category iterator_category;
      const_iterator() : me(0), i( 0 ) { }
      const_iterator(pointer ime, difference_type ii ) : me(ime), i(ii) { }
      const_iterator(const iterator & it) : me(it.me), i(it.i) { }
      const_iterator & operator=( const iterator & it ) { me = it.me; i=it.i;  return *this; }
      const_iterator& operator++() { ++i; return *this; }
      const_iterator operator++( int ) { const_iterator ci = *this; ++i; return ci; }
      const_iterator& operator--() { --i; return *this; }
      const_iterator operator--( int ) { const_iterator ci = *this; --i; return ci; }
      difference_type operator-( const const_iterator & o ) const { return i-o.i; }
      const_iterator operator+( difference_type n ) const { 
	const_iterator ci = *this; ci.i+=n;
	return ci; 
      }
      const_iterator operator-( difference_type n ) const {
        const_iterator ci = *this; ci.i-=n;  
        return ci;
      }
      bool operator<( const const_iterator & o ) { return i<o.i; }
      bool operator==( const const_iterator& ci ) const { return i==ci.i; }
      bool operator!=( const const_iterator& ci ) const { return i!=ci.i; }
      inline reference operator * () const;
      pointer operator->() const { return & ( operator*() ); }
      const_iterator & operator +=( difference_type d ) { i+=d; return *this; }
      const_iterator & operator -=( difference_type d ) { i-=d; return *this; }
    private:
       pointer me;
       difference_type i;
    };

  }
}

#endif
