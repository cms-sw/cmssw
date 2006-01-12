#ifndef Common_ref_collection_h
#define Common_ref_collection_h
#include "FWCore/EDProduct/interface/RefProd.h"

template<typename R, typename C>
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
  
  ref_collection<R, C> & operator=( const ref_collection<R, C> & );
  
  void reserve( size_t );
  void push_back( const value_type & );  
  void clear();

  void setRef( const edm::RefProd<R> & ref ) { ref_ = ref; }
  const edm::RefProd<R> ref() const { return ref_; }

private:
  C data_;
  edm::RefProd<R> ref_;
};

template<typename R, typename C>
  inline ref_collection<R, C>::ref_collection() : data_(), ref_() { 
}

template<typename R, typename C>
  inline ref_collection<R, C>::ref_collection( size_type n ) : data_( n ), ref_() { 
}

template<typename R, typename C>
  inline ref_collection<R, C>::ref_collection( const ref_collection<R, C> & o ) : 
    data_( o.data_ ), ref_( o.ref_ ) { 
}

template<typename R, typename C>
  inline ref_collection<R, C>::~ref_collection() { 
}

template<typename R, typename C>
  inline ref_collection<R, C> & ref_collection<R, C>::operator=( const ref_collection<R, C> & o ) {
  data_ = o.data_;
  ref_ = o.ref_;
  return * this;
}

template<typename R, typename C>
  inline typename ref_collection<R, C>::iterator ref_collection<R, C>::begin() {
  return data_.begin();
}

template<typename R, typename C>
  inline typename ref_collection<R, C>::iterator ref_collection<R, C>::end() {
  return data_.end();
}

template<typename R, typename C>
  inline typename ref_collection<R, C>::const_iterator ref_collection<R, C>::begin() const {
  return data_.begin();
}

template<typename R, typename C>
  inline typename ref_collection<R, C>::const_iterator ref_collection<R, C>::end() const {
  return data_.end();
}

template<typename R, typename C>
  inline typename ref_collection<R, C>::size_type ref_collection<R, C>::size() const {
  return data_.size();
}

template<typename R, typename C>
  inline bool ref_collection<R, C>::empty() const {
  return data_.empty();
}

template<typename R, typename C>
  inline typename ref_collection<R, C>::reference ref_collection<R, C>::operator[]( size_type n ) {
  return data_[ n ];
}

template<typename R, typename C>
  inline typename ref_collection<R, C>::const_reference ref_collection<R, C>::operator[]( size_type n ) const {
  return data_[ n ];
}

template<typename R, typename C>
  inline void ref_collection<R, C>::reserve( size_t n ) {
  data_.reserve( n );
}

template<typename R, typename C>
  inline void ref_collection<R, C>::push_back( const value_type & t ) {
  data_.push_back( t );
}

template<typename R, typename C>
  inline void ref_collection<R, C>::clear() {
  data_.clear();
}

#endif
