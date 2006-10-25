#ifndef Common_OwnVector_h
#define Common_OwnVector_h
// $Id: OwnVector.h,v 1.9 2006/08/31 07:55:23 llista Exp $
#include <vector>
#include "DataFormats/Common/interface/ClonePolicy.h"
#include "DataFormats/Common/interface/traits.h"
#include <algorithm>
#include <functional>

#include "FWCore/Utilities/interface/EDMException.h"

#if defined CMS_USE_DEBUGGING_ALLOCATOR
#include "DataFormats/Common/interface/debugging_allocator.h"
#endif


namespace edm {

  template <typename T, typename P = ClonePolicy<T> >
  class OwnVector  {
  private:
#if defined(CMS_USE_DEBUGGING_ALLOCATOR)
    typedef std::vector<T*, debugging_allocator<T> > base;
#else
    typedef std::vector<T*> base;
#endif

  public:
    typedef typename base::size_type size_type;
    typedef T value_type;
    typedef T * pointer;
    typedef T & reference;
    typedef const T & const_reference;
    typedef P   policy_type;
      
    class iterator;
    class const_iterator {
    public:
      typedef T value_type;
      typedef T * pointer;
      typedef T & reference;
      typedef ptrdiff_t difference_type;
      typedef typename base::const_iterator::iterator_category iterator_category;
      const_iterator( const typename base::const_iterator & it ) : i( it ) { }
      const_iterator( const const_iterator & it ) : i( it.i ) { }
      const_iterator( const iterator & it ) : i( it.i ) { }
      const_iterator() {}
      const_iterator & operator=( const const_iterator & it ) { i = it.i; return *this; }
      const_iterator& operator++() { ++i; return *this; }
      const_iterator operator++( int ) { const_iterator ci = *this; ++i; return ci; }
      const_iterator& operator--() { --i; return *this; }
      const_iterator operator--( int ) { const_iterator ci = *this; --i; return ci; }
      difference_type operator-( const const_iterator & o ) const { return i - o.i; }
      const_iterator operator+( difference_type n ) const { return const_iterator( i + n ); }
      const_iterator operator-( difference_type n ) const { return const_iterator( i - n ); }
      bool operator<( const const_iterator & o ) const { return i < o.i; }
      bool operator==( const const_iterator& ci ) const { return i == ci.i; }
      bool operator!=( const const_iterator& ci ) const { return i != ci.i; }
      const T & operator * () const { return * * i; }
      //    operator const T * () const { return & * * i; }
      const T * operator->() const { return & ( operator*() ); }
      const_iterator & operator +=( difference_type d ) { i += d; return *this; }
      const_iterator & operator -=( difference_type d ) { i -= d; return *this; }
    private:
      typename base::const_iterator i;
    };
    class iterator {
    public:
      typedef T value_type;
      typedef T * pointer;
      typedef T & reference;
      typedef ptrdiff_t difference_type;
      typedef typename base::iterator::iterator_category iterator_category;
      iterator( const typename base::iterator & it ) : i( it ) { }
      iterator( const iterator & it ) : i( it.i ) { }
      iterator() {}
      iterator & operator=( const iterator & it ) { i = it.i; return *this; }
      iterator& operator++() { ++i; return *this; }
      iterator operator++( int ) { iterator ci = *this; ++i; return ci; }
      iterator& operator--() { --i; return *this; }
      iterator operator--( int ) { iterator ci = *this; --i; return ci; }
      difference_type operator-( const iterator & o ) const { return i - o.i; }
      iterator operator+( difference_type n ) const { return iterator( i + n ); }
      iterator operator-( difference_type n ) const { return iterator( i - n ); }
      bool operator<( const iterator & o ) const { return i < o.i; }
      bool operator==( const iterator& ci ) const { return i == ci.i; }
      bool operator!=( const iterator& ci ) const { return i != ci.i; }
      T & operator * () const { return * * i; }
      //    operator T * () const { return & * * i; }
      //T * & get() { return * i; }
      T * operator->() const { return & ( operator*() ); }
      iterator & operator +=( difference_type d ) { i += d; return *this; }
      iterator & operator -=( difference_type d ) { i -= d; return *this; }
    private:
      typename base::iterator i;
      friend const_iterator::const_iterator( const iterator & );
    };
      
    OwnVector();
    OwnVector( size_type );
    OwnVector( const OwnVector & );
    ~OwnVector();
      
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    size_type size() const;
    bool empty() const;
    reference operator[]( size_type );
    const_reference operator[]( size_type ) const;
      
    OwnVector<T, P> & operator=( const OwnVector<T, P> & );
      
    void reserve( size_t );
    template <class D> void push_back( D*& );
    bool is_back_safe() const;
    void pop_back();
    reference back();
    const_reference back() const;
    reference front();
    const_reference front() const;
      
    void clear();
    template<typename S> 
    void sort( S s );
    void sort();

    void swap(OwnVector<T, P> & other);
      
  private:
    void destroy();
    template<typename O>
    struct Ordering {
      Ordering( const O & c ) : comp( c ) { }
      bool operator()( const T * t1, const T * t2 ) const {
	return comp( * t1, * t2 );
      }
    private:
      O comp;
    };
    template<typename O>
    static Ordering<O> ordering( const O & comp ) {
      return Ordering<O>( comp );
    }
    struct deleter {
      void operator()( T & t ) { delete & t; }
    };
    base data_;      
  };
  
  template<typename T, typename P>
  inline OwnVector<T, P>::OwnVector() : data_() { 
  }
  
  template<typename T, typename P>
  inline OwnVector<T, P>::OwnVector( size_type n ) : data_( n ) { 
  }
  
  template<typename T, typename P>
  inline OwnVector<T, P>::OwnVector( const OwnVector<T, P> & o ) : data_( o.size() ) {
    size_type current = 0;
    for ( const_iterator i = o.begin(), e = o.end(); i != e; ++i,++current) 
      data_[current] = policy_type::clone(*i);
  }
  
  template<typename T, typename P>
  inline OwnVector<T, P>::~OwnVector() { 
    destroy();
  }
  
  template<typename T, typename P>
  inline OwnVector<T, P> & OwnVector<T, P>::operator=( const OwnVector<T, P> & o ) {
    OwnVector<T,P> temp(o);
    swap(temp);
    return *this;
  }
  
  template<typename T, typename P>
  inline typename OwnVector<T, P>::iterator OwnVector<T, P>::begin() {
    return iterator( data_.begin() );
  }
  
  template<typename T, typename P>
  inline typename OwnVector<T, P>::iterator OwnVector<T, P>::end() {
    return iterator( data_.end() );
  }
  
  template<typename T, typename P>
  inline typename OwnVector<T, P>::const_iterator OwnVector<T, P>::begin() const {
    return const_iterator( data_.begin() );
  }
  
  template<typename T, typename P>
  inline typename OwnVector<T, P>::const_iterator OwnVector<T, P>::end() const {
    return const_iterator( data_.end() );
  }
  
  template<typename T, typename P>
  inline typename OwnVector<T, P>::size_type OwnVector<T, P>::size() const {
    return data_.size();
  }
  
  template<typename T, typename P>
  inline bool OwnVector<T, P>::empty() const {
    return data_.empty();
  }
  
  template<typename T, typename P>
  inline typename OwnVector<T, P>::reference OwnVector<T, P>::operator[]( size_type n ) {
    //return * data_.operator[]( n );
    return *data_[n];
  }
  
  template<typename T, typename P>
  inline typename OwnVector<T, P>::const_reference OwnVector<T, P>::operator[]( size_type n ) const {
    //return * data_.operator[]( n );
    return *data_[n];
  }
  
  template<typename T, typename P>
  inline void OwnVector<T, P>::reserve( size_t n ) {
    data_.reserve( n );
  }
  
  template<typename T, typename P>
  template<class D>
  inline void OwnVector<T, P>::push_back( D*& t ) {
    data_.push_back( t );
    t = 0;
  }

  template<typename T, typename P>
  inline void OwnVector<T, P>::pop_back() {
    // We have to delete the pointed-to thing, before we squeeze it
    // out of the vector...
    delete data_.back();
    data_.pop_back();
  }

  template <typename T, typename P>
  inline bool OwnVector<T, P>::is_back_safe() const
  {
    return data_.back() != 0;
  }

  template<typename T, typename P>
  inline typename OwnVector<T, P>::reference OwnVector<T, P>::back() {
    T* result = data_.back();
    if (result == 0)
      throw edm::Exception(errors::NullPointerError)
	<< "In OwnVector::back() we have intercepted an attempt to dereference a null pointer\n"
	<< "Since OwnVector is allowed to contain null pointers, you much assure that the\n"
	<< "pointer at the end of the collection is not null before calling back()\n"
	<< "if you wish to avoid this exception.\n"
	<< "Consider using OwnVector::is_back_safe()\n";
    return * data_.back();
  }
  
  template<typename T, typename P>
  inline typename OwnVector<T, P>::const_reference OwnVector<T, P>::back() const {
    T* result = data_.back();
    if (result == 0)
      throw edm::Exception(errors::NullPointerError)
	<< "In OwnVector::back() we have intercepted an attempt to dereference a null pointer\n"
	<< "Since OwnVector is allowed to contain null pointers, you much assure that the\n"
	<< "pointer at the end of the collection is not null before calling back()\n"
	<< "if you wish to avoid this exception.\n"
	<< "Consider using OwnVector::is_back_safe()\n";
    return * data_.back();
  }
  
  template<typename T, typename P>
  inline typename OwnVector<T, P>::reference OwnVector<T, P>::front() {
    return * data_.front();
  }
  
  template<typename T, typename P>
  inline typename OwnVector<T, P>::const_reference OwnVector<T, P>::front() const {
    return * data_.front();
  }
  
  template<typename T, typename P>
  inline void OwnVector<T, P>::destroy() {
    std::for_each( begin(), end(), deleter() );
  }
  
  template<typename T, typename P>
  inline void OwnVector<T, P>::clear() {
    destroy();
    data_.clear();
  }

  template<typename T, typename P> template<typename S>
  void OwnVector<T, P>::sort( S comp ) {
    std::sort( data_.begin(), data_.end(), ordering( comp ) );
  }

  template<typename T, typename P>
  void OwnVector<T, P>::sort() {
    std::sort( data_.begin(), data_.end(), ordering( std::less<value_type>() ) );
  }

  template<typename T, typename P>
  inline void OwnVector<T, P>::swap(OwnVector<T, P>& other) {
    data_.swap(other.data_);
  }
    
  template<typename T, typename P>
  inline void swap(OwnVector<T, P>& a, OwnVector<T, P>& b) {
    a.swap(b);
  }

  /// has swap function
  template<typename T, typename P>
  struct has_swap<edm::OwnVector<T,P> > {
    static bool const value = true;
  };
}

#endif
