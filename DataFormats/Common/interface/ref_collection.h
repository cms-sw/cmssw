#ifndef Common_ref_collection_h
#define Common_ref_collection_h
#include "DataFormats/Common/interface/collection_base.h"
#include "FWCore/EDProduct/interface/RefProd.h"
#include "FWCore/EDProduct/interface/Ref.h"

template<typename C, typename R>
class ref_collection : public collection_base<C> {
public:
  typedef typename C::size_type size_type;
  typedef typename C::value_type value_type;
  typedef typename C::reference reference;
  typedef typename C::pointer pointer;
  typedef typename C::const_reference const_reference;
  typedef typename C::iterator iterator;
  typedef typename C::const_iterator const_iterator;
  ref_collection();
  ref_collection( size_type );
  ref_collection( const ref_collection & );
  ~ref_collection();
 
  ref_collection<C, R> & operator=( const ref_collection<C, R> & );

  void setRef( const edm::RefProd<R> & ref ) { ref_ = ref; }
  const edm::RefProd<R> & ref() const { return ref_; }

private:
  C data_;
  edm::RefProd<R> ref_;
};

template<typename C, typename R>
  inline ref_collection<C, R>::ref_collection() : collection_base<C>() { 
}

template<typename C, typename R>
  inline ref_collection<C, R>::ref_collection( size_type n ) : collection_base<C>( n ) { 
}

template<typename C, typename R>
  inline ref_collection<C, R>::ref_collection( const ref_collection<C, R> & o ) : 
    collection_base<C>( o ), ref_( o.ref_ ) { 
}

template<typename C, typename R>
  inline ref_collection<C, R>::~ref_collection() { 
}

template<typename C, typename R>
  inline ref_collection<C, R> & ref_collection<C, R>::operator=( const ref_collection<C, R> & o ) {
  collection_base<C>::operator=( o );
  ref_ = o.ref_;
  return * this;
}

#include "DataFormats/Common/interface/getAssociated.h"

#endif
