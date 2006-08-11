#ifndef Common_ext_collection_h
#define Common_ext_collection_h

#include "DataFormats/Common/interface/traits.h"
#include <algorithm>

namespace edm {

  template<typename C, typename Ext>
  class ExtCollection {
  public:
    typedef typename C::size_type size_type;
    typedef typename C::value_type value_type;
    typedef typename C::reference reference;
    typedef typename C::pointer pointer;
    typedef typename C::const_reference const_reference;
    typedef typename C::iterator iterator;
    typedef typename C::const_iterator const_iterator;
    ExtCollection();
    ExtCollection( size_type );
    ExtCollection( const ExtCollection & );
    ~ExtCollection();
    
    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;
    size_type size() const;
    bool empty() const;
    reference operator[]( size_type );
    const_reference operator[]( size_type ) const;
    
    ExtCollection<C, Ext> & operator=( const ExtCollection<C, Ext> & );
    
    void reserve( size_type );
    void push_back( const value_type & );  
    void clear();
    void swap(ExtCollection<C, Ext> & other);
    Ext & ext() { return ext_; }
    const Ext & ext() const { return ext_; }
  private:
    C data_;
    Ext ext_;
  };
  
  template<typename C, typename Ext>
  inline ExtCollection<C, Ext>::ExtCollection() : data_(), ext_() { 
  }
  
  template<typename C, typename Ext>
  inline ExtCollection<C, Ext>::ExtCollection( size_type n ) : data_( n ), ext_() { 
  }
  
  template<typename C, typename Ext>
  inline ExtCollection<C, Ext>::ExtCollection( const ExtCollection<C, Ext> & o ) : 
    data_( o.data_ ), ext_( o.ext_ ) { 
  }
  
  template<typename C, typename Ext>
  inline ExtCollection<C, Ext>::~ExtCollection() { 
  }
  
  template<typename C, typename Ext>
  inline ExtCollection<C, Ext> & ExtCollection<C, Ext>::operator=( const ExtCollection<C, Ext> & o ) {
    data_ = o.data_;
    ext_ = o.ext_;
    return * this;
  }
  
  template<typename C, typename Ext>
  inline typename ExtCollection<C, Ext>::iterator ExtCollection<C, Ext>::begin() {
    return data_.begin();
  }
  
  template<typename C, typename Ext>
  inline typename ExtCollection<C, Ext>::iterator ExtCollection<C, Ext>::end() {
    return data_.end();
  }
  
  template<typename C, typename Ext>
  inline typename ExtCollection<C, Ext>::const_iterator ExtCollection<C, Ext>::begin() const {
    return data_.begin();
  }
  
  template<typename C, typename Ext>
  inline typename ExtCollection<C, Ext>::const_iterator ExtCollection<C, Ext>::end() const {
    return data_.end();
  }
  
  template<typename C, typename Ext>
  inline typename ExtCollection<C, Ext>::size_type ExtCollection<C, Ext>::size() const {
    return data_.size();
  }
  
  template<typename C, typename Ext>
  inline bool ExtCollection<C, Ext>::empty() const {
    return data_.empty();
  }
  
  template<typename C, typename Ext>
  inline typename ExtCollection<C, Ext>::reference ExtCollection<C, Ext>::operator[]( size_type n ) {
    return data_[ n ];
  }
  
  template<typename C, typename Ext>
  inline typename ExtCollection<C, Ext>::const_reference ExtCollection<C, Ext>::operator[]( size_type n ) const {
    return data_[ n ];
  }
  
  template<typename C, typename Ext>
  inline void ExtCollection<C, Ext>::reserve( size_type n ) {
    data_.reserve( n );
  }
  
  template<typename C, typename Ext>
  inline void ExtCollection<C, Ext>::push_back( const value_type & t ) {
    data_.push_back( t );
  }
  
  template<typename C, typename Ext>
  inline void ExtCollection<C, Ext>::clear() {
    data_.clear();
    ext_ = Ext();
  }

  template<typename C, typename Ext> inline void swap( ExtCollection<C, Ext> & a, ExtCollection<C, Ext> & b );

  template<typename C, typename Ext>
  inline void ExtCollection<C, Ext>::swap( ExtCollection<C, Ext> & other ) {
    using edm::swap;
    swap(data_, other.data_);
    swap(ext_, other.ext_);
  }

  template<typename C, typename Ext>
  inline void swap( ExtCollection<C, Ext> & a, ExtCollection<C, Ext> & b ) {
    a.swap(b);
  }

  // has swap function
  template<typename C, typename Ext>
  struct has_swap<edm::ExtCollection<C, Ext> > {
    static bool const value = true;
  };

}

#endif
