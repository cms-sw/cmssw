#ifndef Candidate_const_iterator_imp_specific_h
#define Candidate_const_iterator_imp_specific_h

#include "DataFormats/Candidate/interface/iterator_deref.h"
#include "DataFormats/Candidate/interface/const_iterator_imp.h"
#include<boost/static_assert.hpp>

namespace reco {
  namespace candidate {
    template<typename C>
    struct const_iterator_imp_specific : public const_iterator_imp {
    private:
      typedef typename C::const_iterator const_iterator;
    public:
      typedef ptrdiff_t difference_type;
      const_iterator_imp_specific() { }
      explicit const_iterator_imp_specific( const const_iterator & it ) : i ( it ) { }
      ~const_iterator_imp_specific() { }
      const_iterator_imp_specific * clone() const { return new const_iterator_imp_specific<C>( i ); }
      void increase() { ++i; }
      void decrease() { --i; }
      void increase( difference_type d ) { i += d; }
      void decrease( difference_type d ) { i -= d; }
      bool equal_to( const const_iterator_imp * o ) const { return i == dc( o ); }
      bool less_than( const const_iterator_imp * o ) const { return i < dc( o ); }
      void assign( const const_iterator_imp * o ) { i = dc( o ); }
      const Candidate & deref() const { return iterator_deref<C>::deref(i); }
      difference_type difference( const const_iterator_imp * o ) const { return i - dc( o ); }
    private:
      const const_iterator & dc( const const_iterator_imp * o ) const {
	return dynamic_cast<const const_iterator_imp_specific *>( o )->i;
      }
      const_iterator & dc( const_iterator_imp * o ) const {
	return dynamic_cast<const_iterator_imp_specific *>( o )->i;
      }
      const_iterator i;
    };

    template<typename C>
    struct const_iterator_imp_specific_dummy : public const_iterator_imp {
      typedef ptrdiff_t difference_type;
      const_iterator_imp_specific_dummy() { }
      ~const_iterator_imp_specific_dummy() { }
      const_iterator_imp_specific_dummy * clone() const { return new const_iterator_imp_specific_dummy<C>; }
      void increase() { }
      void decrease() { }
      void increase( difference_type d ) { }
      void decrease( difference_type d ) { }
      bool equal_to( const const_iterator_imp * o ) const { return true; }
      bool less_than( const const_iterator_imp * o ) const { return false; }
      void assign( const const_iterator_imp * o ) {  }
      const Candidate & deref() const { 
	throw cms::Exception("Invalid Dereference") 
	  << "can't dereference an interator for a Candidate with read-only"
	  << "references o daughters";
      }
      difference_type difference( const const_iterator_imp * o ) const { return 0; }
    };

  }
}

#endif
