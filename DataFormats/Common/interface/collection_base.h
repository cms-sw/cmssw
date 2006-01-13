#ifndef Common_collection_base_h
#define Common_collection_base_h

template<typename C>
class collection_base {
public:
  typedef typename C::size_type size_type;
  typedef typename C::value_type value_type;
  typedef typename C::reference reference;
  typedef typename C::pointer pointer;
  typedef typename C::const_reference const_reference;
  typedef typename C::iterator iterator;
  typedef typename C::const_iterator const_iterator;
  collection_base();
  collection_base( size_type );
  collection_base( const collection_base & );
  ~collection_base();
  
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  size_type size() const;
  bool empty() const;
  reference operator[]( size_type );
  const_reference operator[]( size_type ) const;
  
  collection_base<C> & operator=( const collection_base<C> & );
  
  void reserve( size_t );
  void push_back( const value_type & );  
  void clear();

private:
  C data_;
};

template<typename C>
  inline collection_base<C>::collection_base() : data_() { 
}

template<typename C>
  inline collection_base<C>::collection_base( size_type n ) : data_( n ) { 
}

template<typename C>
  inline collection_base<C>::collection_base( const collection_base<C> & o ) : 
    data_( o.data_ ) { 
}

template<typename C>
  inline collection_base<C>::~collection_base() { 
}

template<typename C>
  inline collection_base<C> & collection_base<C>::operator=( const collection_base<C> & o ) {
  data_ = o.data_;
  return * this;
}

template<typename C>
  inline typename collection_base<C>::iterator collection_base<C>::begin() {
  return data_.begin();
}

template<typename C>
  inline typename collection_base<C>::iterator collection_base<C>::end() {
  return data_.end();
}

template<typename C>
  inline typename collection_base<C>::const_iterator collection_base<C>::begin() const {
  return data_.begin();
}

template<typename C>
  inline typename collection_base<C>::const_iterator collection_base<C>::end() const {
  return data_.end();
}

template<typename C>
  inline typename collection_base<C>::size_type collection_base<C>::size() const {
  return data_.size();
}

template<typename C>
  inline bool collection_base<C>::empty() const {
  return data_.empty();
}

template<typename C>
  inline typename collection_base<C>::reference collection_base<C>::operator[]( size_type n ) {
  return data_[ n ];
}

template<typename C>
  inline typename collection_base<C>::const_reference collection_base<C>::operator[]( size_type n ) const {
  return data_[ n ];
}

template<typename C>
  inline void collection_base<C>::reserve( size_t n ) {
  data_.reserve( n );
}

template<typename C>
  inline void collection_base<C>::push_back( const value_type & t ) {
  data_.push_back( t );
}

template<typename C>
  inline void collection_base<C>::clear() {
  data_.clear();
}

#endif
