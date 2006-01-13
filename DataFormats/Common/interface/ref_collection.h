#ifndef Common_ref_collection_h
#define Common_ref_collection_h
#include "FWCore/EDProduct/interface/RefProd.h"
#include "FWCore/EDProduct/interface/Ref.h"

template<typename C, typename R>
class ref_collection {
public:
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
  
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  size_type size() const;
  bool empty() const;
  reference operator[]( size_type );
  const_reference operator[]( size_type ) const;
  
  ref_collection<C, R> & operator=( const ref_collection<C, R> & );
  
  void reserve( size_t );
  void push_back( const value_type & );  
  void clear();

  void setRef( const edm::RefProd<R> & ref ) { ref_ = ref; }
  const edm::RefProd<R> ref() const { return ref_; }

private:
  C data_;
  edm::RefProd<R> ref_;
};

template<typename C, typename R>
const typename R::value_type & getAssociated( const  edm::Ref<ref_collection<C, R> > & ref ) {
  // the following could be a method of edm::Ref
  const edm::EDProduct * edp = ref.productGetter()->getIt( ref.id() );
  const edm::Wrapper<ref_collection<C, R> > * w = 
    dynamic_cast<const edm::Wrapper<ref_collection<C, R> > *>( edp );
  const ref_collection<C, R> * coll = w->product();
  // 
  edm::RefProd<R> assocCollRef = coll->ref();
  const typename R::value_type & ret = (*assocCollRef)[ ref.index() ];
  return ret;
}

template<typename C, typename R>
  inline ref_collection<C, R>::ref_collection() : data_(), ref_() { 
}

template<typename C, typename R>
  inline ref_collection<C, R>::ref_collection( size_type n ) : data_( n ), ref_() { 
}

template<typename C, typename R>
  inline ref_collection<C, R>::ref_collection( const ref_collection<C, R> & o ) : 
    data_( o.data_ ), ref_( o.ref_ ) { 
}

template<typename C, typename R>
  inline ref_collection<C, R>::~ref_collection() { 
}

template<typename C, typename R>
  inline ref_collection<C, R> & ref_collection<C, R>::operator=( const ref_collection<C, R> & o ) {
  data_ = o.data_;
  ref_ = o.ref_;
  return * this;
}

template<typename C, typename R>
  inline typename ref_collection<C, R>::iterator ref_collection<C, R>::begin() {
  return data_.begin();
}

template<typename C, typename R>
  inline typename ref_collection<C, R>::iterator ref_collection<C, R>::end() {
  return data_.end();
}

template<typename C, typename R>
  inline typename ref_collection<C, R>::const_iterator ref_collection<C, R>::begin() const {
  return data_.begin();
}

template<typename C, typename R>
  inline typename ref_collection<C, R>::const_iterator ref_collection<C, R>::end() const {
  return data_.end();
}

template<typename C, typename R>
  inline typename ref_collection<C, R>::size_type ref_collection<C, R>::size() const {
  return data_.size();
}

template<typename C, typename R>
  inline bool ref_collection<C, R>::empty() const {
  return data_.empty();
}

template<typename C, typename R>
  inline typename ref_collection<C, R>::reference ref_collection<C, R>::operator[]( size_type n ) {
  return data_[ n ];
}

template<typename C, typename R>
  inline typename ref_collection<C, R>::const_reference ref_collection<C, R>::operator[]( size_type n ) const {
  return data_[ n ];
}

template<typename C, typename R>
  inline void ref_collection<C, R>::reserve( size_t n ) {
  data_.reserve( n );
}

template<typename C, typename R>
  inline void ref_collection<C, R>::push_back( const value_type & t ) {
  data_.push_back( t );
}

template<typename C, typename R>
  inline void ref_collection<C, R>::clear() {
  data_.clear();
}

#endif
