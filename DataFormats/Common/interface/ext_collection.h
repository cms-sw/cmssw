#ifndef Common_ext_collection_h
#define Common_ext_collection_h

template<typename C, typename Ext>
class ext_collection {
public:
  typedef typename C::size_type size_type;
  typedef typename C::value_type value_type;
  typedef typename C::reference reference;
  typedef typename C::pointer pointer;
  typedef typename C::const_reference const_reference;
  typedef typename C::iterator iterator;
  typedef typename C::const_iterator const_iterator;
  ext_collection();
  ext_collection( size_type );
  ext_collection( const ext_collection & );
  ~ext_collection();
  
  iterator begin();
  iterator end();
  const_iterator begin() const;
  const_iterator end() const;
  size_type size() const;
  bool empty() const;
  reference operator[]( size_type );
  const_reference operator[]( size_type ) const;
  
  ext_collection<C, Ext> & operator=( const ext_collection<C, Ext> & );
  
  void reserve( size_t );
  void push_back( const value_type & );  
  void clear();
  Ext & ext() { return ext_; }
  const Ext & ext() const { return ext_; }
private:
  C data_;
  Ext ext_;
};

template<typename C, typename Ext>
  inline ext_collection<C, Ext>::ext_collection() : data_(), ext_() { 
}

template<typename C, typename Ext>
  inline ext_collection<C, Ext>::ext_collection( size_type n ) : data_( n ), ext_() { 
}

template<typename C, typename Ext>
  inline ext_collection<C, Ext>::ext_collection( const ext_collection<C, Ext> & o ) : 
    data_( o.data_ ), ext_( o.ext_ ) { 
}

template<typename C, typename Ext>
  inline ext_collection<C, Ext>::~ext_collection() { 
}

template<typename C, typename Ext>
  inline ext_collection<C, Ext> & ext_collection<C, Ext>::operator=( const ext_collection<C, Ext> & o ) {
  data_ = o.data_;
  ext_ = o.ext_;
  return * this;
}

template<typename C, typename Ext>
  inline typename ext_collection<C, Ext>::iterator ext_collection<C, Ext>::begin() {
  return data_.begin();
}

template<typename C, typename Ext>
  inline typename ext_collection<C, Ext>::iterator ext_collection<C, Ext>::end() {
  return data_.end();
}

template<typename C, typename Ext>
  inline typename ext_collection<C, Ext>::const_iterator ext_collection<C, Ext>::begin() const {
  return data_.begin();
}

template<typename C, typename Ext>
  inline typename ext_collection<C, Ext>::const_iterator ext_collection<C, Ext>::end() const {
  return data_.end();
}

template<typename C, typename Ext>
  inline typename ext_collection<C, Ext>::size_type ext_collection<C, Ext>::size() const {
  return data_.size();
}

template<typename C, typename Ext>
  inline bool ext_collection<C, Ext>::empty() const {
  return data_.empty();
}

template<typename C, typename Ext>
  inline typename ext_collection<C, Ext>::reference ext_collection<C, Ext>::operator[]( size_type n ) {
  return data_[ n ];
}

template<typename C, typename Ext>
  inline typename ext_collection<C, Ext>::const_reference ext_collection<C, Ext>::operator[]( size_type n ) const {
  return data_[ n ];
}

template<typename C, typename Ext>
  inline void ext_collection<C, Ext>::reserve( size_t n ) {
  data_.reserve( n );
}

template<typename C, typename Ext>
  inline void ext_collection<C, Ext>::push_back( const value_type & t ) {
  data_.push_back( t );
}

template<typename C, typename Ext>
  inline void ext_collection<C, Ext>::clear() {
  data_.clear();
  ext_ = Ext();
}

#endif
