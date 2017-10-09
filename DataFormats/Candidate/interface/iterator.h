#ifndef Candidate_iterator_h
#define Candidate_iterator_h

/* \class reco::candidate::iterator
 *
 * \author Luca Lista, INFN
 *
 */
#include "DataFormats/Candidate/interface/CandidateFwd.h"

namespace reco {
  namespace candidate {
    struct const_iterator;
    struct iterator {
      typedef Candidate value_type;
      typedef Candidate * pointer;
      typedef Candidate & reference;
      typedef ptrdiff_t difference_type;
      typedef std::vector<int>::iterator::iterator_category iterator_category;
      iterator() : me(0), i( 0 ) { }
      iterator(pointer ime, difference_type ii ) : me(ime), i(ii) { }
      iterator& operator++() { ++i; return *this; }
      iterator operator++( int ) { iterator ci = *this; ++i; return ci; }
      iterator& operator--() { --i; return *this; }
      iterator operator--( int ) { iterator ci = *this; --i; return ci; }
      difference_type operator-( const iterator & o ) const { return i-o.i; }
      iterator operator+( difference_type n ) const { 
	iterator ci = *this; ci.i+=n;
	return ci; 
      }
      iterator operator-( difference_type n ) const {
        iterator ci = *this; ci.i-=n;  
        return ci;
      }
      bool operator<( const iterator & o ) { return i<o.i; }
      bool operator==( const iterator& ci ) const { return i==ci.i; }
      bool operator!=( const iterator& ci ) const { return i!=ci.i; }

      inline reference operator * () const;
      pointer operator->() const { return & ( operator*() ); }
      iterator & operator +=( difference_type d ) { i+=d; return *this; }
      iterator & operator -=( difference_type d ) { i-=d; return *this; }
    private:
       pointer me;
       difference_type i;
       friend struct const_iterator;
    };

  }
}

#endif
