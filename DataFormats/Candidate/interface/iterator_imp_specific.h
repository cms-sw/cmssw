#ifndef Candidate_iterator_imp_specific_h
#define Candidate_iterator_imp_specific_h

#include "DataFormats/Candidate/interface/iterator_deref.h"
#include "DataFormats/Candidate/interface/const_iterator_imp_specific.h"
#include "DataFormats/Candidate/interface/iterator_imp.h"
#include "FWCore/Utilities/interface/Exception.h"

namespace reco {
  namespace candidate {
   
    template<typename C>
    struct iterator_imp_specific : public iterator_imp  {
    private:
      typedef typename C::iterator iterator;
    public:
      typedef ptrdiff_t difference_type;
      iterator_imp_specific() { }
      explicit iterator_imp_specific( const iterator & it ) : i ( it ) { }
      ~iterator_imp_specific() { }
      iterator_imp_specific * clone() const { return new iterator_imp_specific<C>( i ); }
      const_iterator_imp_specific<C> * const_clone() const { return new const_iterator_imp_specific<C>( i ); }
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
      const iterator & dc( const iterator_imp * o ) const {
	return dynamic_cast<const iterator_imp_specific *>( o )->i;
      }
      iterator & dc( iterator_imp * o ) const {
	return dynamic_cast<iterator_imp_specific *>( o )->i;
      }
      iterator i;
    };

    template<typename C>
    struct iterator_imp_specific_dummy : public iterator_imp {
      typedef ptrdiff_t difference_type;
      iterator_imp_specific_dummy() { }
      ~iterator_imp_specific_dummy() { }
      iterator_imp_specific_dummy * clone() const { return new iterator_imp_specific_dummy; }
      const_iterator_imp_specific_dummy<C> * const_clone() const { return new const_iterator_imp_specific_dummy<C>; }
      void increase() { }
      void decrease() { }
      void increase( difference_type d ) { }
      void decrease( difference_type d ) { }
      bool equal_to( const iterator_imp * o ) const { return true; }
      bool less_than( const iterator_imp * o ) const { return false; }
      void assign( const iterator_imp * o ) { }
      Candidate & deref() const { 
	throw cms::Exception("Invalid Dereference") 
	  << "can't dereference an interator for a Candidate with read-only"
	  << "references o daughters";
      }
      difference_type difference( const iterator_imp * o ) const { return 0; }
    };

 }
}

#endif
