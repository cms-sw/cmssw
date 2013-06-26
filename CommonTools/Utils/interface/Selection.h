#ifndef CommonTools_Utils_Selection_h
#define CommonTools_Utils_Selection_h
#include <vector>

template<typename C, 
	 typename Selector,
	 typename StoreContainer = std::vector<const typename C::value_type *> >
class Selection {
public:
  typedef typename C::value_type value_type;
  typedef typename C::size_type size_type;
  typedef value_type & reference;
  typedef const value_type & const_reference;
  Selection( const C & c, const Selector & sel ) :
    select_( sel ) {
    for( typename C::const_iterator i = c.begin(); i != c.end(); ++i ) {
      if ( select_( *i ) ) selected_.push_back( & * i );
    }
  }
  class const_iterator {
  public:
    typedef typename Selection<C,Selector,StoreContainer>::value_type value_type;
    typedef value_type * pointer;
    typedef value_type & reference;
    typedef std::ptrdiff_t difference_type;
    typedef typename StoreContainer::const_iterator::iterator_category iterator_category;
    const_iterator(const typename StoreContainer::const_iterator & it) : i(it) { }
    const_iterator(const const_iterator & it) : i(it.i) { }
    const_iterator() {}
    const_iterator & operator=(const const_iterator & it) { i = it.i; return *this; }
    const_iterator& operator++() { ++i; return *this; }
    const_iterator operator++(int) { const_iterator ci = *this; ++i; return ci; }
    const_iterator& operator--() { --i; return *this; }
    const_iterator operator--(int) { const_iterator ci = *this; --i; return ci; }
    difference_type operator-(const const_iterator & o) const { return i - o.i; }
    const_iterator operator+(difference_type n) const { return const_iterator(i + n); }
    const_iterator operator-(difference_type n) const { return const_iterator(i - n); }
    bool operator<(const const_iterator & o) const { return i < o.i; }
    bool operator==(const const_iterator& ci) const { return i == ci.i; }
    bool operator!=(const const_iterator& ci) const { return i != ci.i; }
    const value_type & operator *() const { return **i; }
    const value_type * operator->() const { return & (operator*()); }
    const_iterator & operator +=(difference_type d) { i += d; return *this; }
    const_iterator & operator -=(difference_type d) { i -= d; return *this; }
  private:
    typename StoreContainer::const_iterator i;
  };
  const_iterator begin() const { return const_iterator( selected_.begin() ); }
  const_iterator end() const { return const_iterator( selected_.end() ); }
  size_type size() const { return selected_.size(); }
  bool empty() const { return selected_.empty(); }
  const_reference operator[]( size_type i ) { return * selected_[i]; }
private:
  Selector select_;
  StoreContainer selected_;
};

#endif
