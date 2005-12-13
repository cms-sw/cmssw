#ifndef PHYSICSTOOLS_OWN_VECTOR_H
#define PHYSICSTOOLS_OWN_VECTOR_H
// $Id: own_vector.h,v 1.5 2005/12/13 01:44:59 llista Exp $
#include <vector>
#include <algorithm>
#include "PhysicsTools/Utilities/interface/ClonePolicy.h"

template <typename T, typename P = ClonePolicy<T> >
class own_vector  {
  private:
  typedef std::vector<T*> base;
  public:
  typedef typename base::size_type size_type;
  typedef T value_type;
  typedef T & reference;
  typedef const T & const_reference;
  struct iterator {
    typedef T value_type;
    typedef T * pointer;
    typedef T & reference;
    typedef ptrdiff_t difference_type;
    typedef typename base::iterator::iterator_category iterator_category;
    iterator( const typename base::iterator & it ) : i( it ) { }
    iterator( const iterator & it ) : i( it.i ) { }
    iterator & operator=( const iterator & it ) { i = it.i; return *this; }
    iterator& operator++() { ++i; return *this; }
    iterator operator++( int ) { iterator ci = *this; ++i; return ci; }
    bool operator==( const iterator& ci ) const { return i == ci.i; }
    bool operator!=( const iterator& ci ) const { return i != ci.i; }
    T & operator * () const { return * * i; }
    operator T * () const { return & * * i; }
    T * & get() { return * i; }
    T * operator->() const { return & ( operator*() ); }
    iterator & operator +=( difference_type d ) { i += d; return *this; }
    iterator & operator -=( difference_type d ) { i -= d; return *this; }
  private:
    typename base::iterator i;
  };
  struct const_iterator {
    typedef T value_type;
    typedef T * pointer;
    typedef T & reference;
    typedef ptrdiff_t difference_type;
    typedef typename base::const_iterator::iterator_category iterator_category;
    const_iterator( const typename base::const_iterator & it ) : i( it ) { }
    const_iterator( const const_iterator & it ) : i( it.i ) { }
    const_iterator & operator=( const const_iterator & it ) { i = it.i; return *this; }
    const_iterator& operator++() { ++i; return *this; }
    const_iterator operator++( int ) { const_iterator ci = *this; ++i; return ci; }
    bool operator==( const const_iterator& ci ) const { return i == ci.i; }
    bool operator!=( const const_iterator& ci ) const { return i != ci.i; }
    const T & operator * () const { return * * i; }
    operator const T * () const { return & * * i; }
    const T * operator->() const { return & ( operator*() ); }
    const_iterator & operator +=( difference_type d ) { i += d; return *this; }
    const_iterator & operator -=( difference_type d ) { i -= d; return *this; }
  private:
    typename base::const_iterator i;
  };
  own_vector();
  own_vector( size_type );
  own_vector( const own_vector & );
  ~own_vector();
  
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  size_type size() const;
  bool empty() const;
  reference operator[]( size_type );
  const_reference operator[]( size_type ) const;
  
  own_vector<T, P> & operator=( const own_vector<T, P> & );
  
  void reserve( size_t );
  void push_back( T * );
  
  void clearAndDestroy();

private:
  void destroy();
  void clone( const_iterator, const_iterator, iterator );
  struct deleter {
    void operator()( T & t ) { delete & t; }
  };
  std::vector<T*> data_;
};

template<typename T, typename P>
  inline own_vector<T, P>::own_vector() : data_() { 
}

template<typename T, typename P>
  inline own_vector<T, P>::own_vector( size_type n ) : data_( n ) { 
}

template<typename T, typename P>
  void inline own_vector<T, P>::clone( const_iterator b, const_iterator e, iterator c ) {
  const_iterator i = b;
  iterator j = c;
  for( ; i != e; ++i, ++j ) j.get() = P::clone( * i );
}

template<typename T, typename P>
  inline own_vector<T, P>::own_vector( const own_vector<T, P> & o ) : data_( o.size() ) { 
  clone( o.begin(), o.end(), begin() );
}

template<typename T, typename P>
  inline own_vector<T, P>::~own_vector() { 
  destroy();
}

template<typename T, typename P>
  inline own_vector<T, P> & own_vector<T, P>::operator=( const own_vector<T, P> & o ) {
  destroy();
  data_.resize( o.size() );
  clone( o.begin(), o.end(), begin() );
  return * this;
}

template<typename T, typename P>
  inline typename own_vector<T, P>::iterator own_vector<T, P>::begin() {
  return iterator( data_.begin() );
}

template<typename T, typename P>
  inline typename own_vector<T, P>::iterator own_vector<T, P>::end() {
  return iterator( data_.end() );
}

template<typename T, typename P>
  inline typename own_vector<T, P>::const_iterator own_vector<T, P>::begin() const {
  return const_iterator( data_.begin() );
}

template<typename T, typename P>
  inline typename own_vector<T, P>::const_iterator own_vector<T, P>::end() const {
  return const_iterator( data_.end() );
}

template<typename T, typename P>
  inline typename own_vector<T, P>::size_type own_vector<T, P>::size() const {
  return data_.size();
}

template<typename T, typename P>
  inline bool own_vector<T, P>::empty() const {
  return data_.empty();
}

template<typename T, typename P>
  inline typename own_vector<T, P>::reference own_vector<T, P>::operator[]( size_type n ) {
  return * data_.operator[]( n );
}

template<typename T, typename P>
  inline typename own_vector<T, P>::const_reference own_vector<T, P>::operator[]( size_type n ) const {
  return * data_.operator[]( n );
}

template<typename T, typename P>
  inline void own_vector<T, P>::reserve( size_t n ) {
  data_.reserve( n );
}

template<typename T, typename P>
  inline void own_vector<T, P>::push_back( T * t ) {
  data_.push_back( t );
}

template<typename T, typename P>
  inline void own_vector<T, P>::destroy() {
  std::for_each( begin(), end(), deleter() );
}

template<typename T, typename P>
  inline void own_vector<T, P>::clearAndDestroy() {
  destroy();
  data_.clear();
}

#endif
