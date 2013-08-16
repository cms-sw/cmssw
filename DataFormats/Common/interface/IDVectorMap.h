#ifndef DataFormats_Common_IDVectorMap_h
#define DataFormats_Common_IDVectorMap_h
#include <map>

namespace edm {

  template<typename ID, typename C, typename P>
  class IDVectorMap {
  public:
    typedef typename C::value_type value_type;
    typedef typename C::const_iterator container_iterator;
    typedef std::map<ID, C> map;
    typedef typename map::const_iterator map_iterator;
    struct const_iterator {
      typedef typename IDVectorMap::value_type value_type;
      typedef value_type * pointer;
      typedef value_type & reference;
      typedef typename map_iterator::iterator_category iterator_category;
      const_iterator() { }
      const_iterator(const map_iterator & e, const map_iterator & m, const container_iterator & c) : 
	im(m), em(e), ic(c) {
      }
      const_iterator(const map_iterator & e) : 
	im(e), em(e) {
      }
      const_iterator & operator=(const const_iterator & it) { 
	im = it.im; em = it.em; ic = it.ic;
	return *this; 
      }
      const_iterator& operator++() { 
	++ic;
	while (ic == im->second.end()) {
	  ++im; 
	  if (im == em) return *this;
	  ic = im->second.begin();
	}
	return *this; 
      }
      const_iterator operator++(int) { const_iterator ci = *this; operator++(); return ci; }
      const_iterator& operator--() { 
	if (im == em) { --im; ic = im->second.end(); }
	while (ic == im->second.begin()) {
	  --im; 
	  ic = im->second.end();
	}
	--ic;
	return *this; 
      }
      const_iterator operator--(int) { const_iterator ci = *this; operator--(); return ci; }
      bool operator==(const const_iterator& ci) const { 
	if (im == em && ci.im == im && ci.em == em) return true;
	return im == ci.im && ic == ci.ic; 
      }
      bool operator!=(const const_iterator& ci) const { return ! operator==(ci); }
      const value_type & operator *() const { return *ic; }
    private:
      map_iterator im, em;
      container_iterator ic;
    };

    IDVectorMap() { }
    const_iterator begin() const {
      return const_iterator(map_.end(), map_.begin(), map_.begin()->second.begin());
    }
    const_iterator end() const {
      return const_iterator(map_.end());
    }
    void insert(ID id, const value_type & t) {
      map_[ id ].push_back(P::clone(t));
    }
    template<typename CI>
    void insert(ID id, CI begin, CI end) {
      C & c = map_[ id ];
      for(CI i = begin; i != end; ++i)
	c.push_back(P::clone(*i));
    }

    struct range {
      range (const container_iterator & b, const container_iterator & e) :
	begin(b), end(e) { }
      container_iterator begin, end;
    };
    range get(ID id) const {
      container_iterator begin, end;
      map_iterator i = map_.find(id);
      if (i != map_.end()) { 
	begin = i->second.begin();
	end = i->second.end();
      } else {
	begin = end;
      }
      return range(begin, end);
    }

    template<typename M>
    struct match_iterator {
      typedef typename IDVectorMap::value_type value_type;
      typedef value_type * pointer;
      typedef value_type & reference;
      typedef typename map_iterator::iterator_category iterator_category;
      match_iterator() { }
      match_iterator(const M & ma, const map_iterator & e, const map_iterator & m, const container_iterator & c) : 
	match(ma), im(m), em(e), ic(c) {
      }
      match_iterator(const M & ma, const map_iterator & e) : 
	match(ma), im(e), em(e) {
      }
      match_iterator & operator=(const match_iterator & it) { 
	match = it.match; im = it.im; em = it.em; ic = it.ic;
	return *this; 
      }
      match_iterator& operator++() { 
	++ic;
	while (ic == im->second.end()) {
	  do { ++im; } while (! match(im->first) && im != em);
	  if (im == em) return *this;
	  ic = im->second.begin();
	}
	return *this; 
      }
      match_iterator operator++(int) { match_iterator ci = *this; operator++(); return ci; }
      bool operator==(const match_iterator& ci) const { 
	if (im == em && ci.im == im && ci.em == em) return true;
	return im == ci.im && ic == ci.ic; 
      }
      bool operator!=(const match_iterator& ci) const { return ! operator==(ci); }
      const value_type & operator * () const { return *ic; }
    private:
      M match;
      map_iterator im, em;
      container_iterator ic;
    };

    template<typename M>
    match_iterator<M> begin(const M & m) const {
      return match_iterator<M>(m, map_.end(), map_.begin(), map_.begin()->second.begin());
    }
    template<typename M>
    match_iterator<M> end(const M & m) const {
      return match_iterator<M>(m, map_.end());
    }

    struct id_iterator {
      typedef ID value_type;
      typedef ID * pointer;
      typedef ID & reference;
      typedef typename map_iterator::iterator_category iterator_category;
      id_iterator() { }
      id_iterator(map_iterator o) : i(o) { }
      id_iterator & operator=(const id_iterator & it) { i = it.i; return *this; }
      id_iterator& operator++() { ++i; return *this; }
      id_iterator operator++(int) { id_iterator ci = *this; ++i; return ci; }
      id_iterator& operator--() { --i; return *this; }
      id_iterator operator--(int) { id_iterator ci = *this; --i; return ci; }
      bool operator==(const id_iterator& ci) const { return i == ci.i; }
      bool operator!=(const id_iterator& ci) const { return i != ci.i; }
      const ID operator *() const { return i->first; }
    private:
      map_iterator i;
    };
    id_iterator id_begin() const { return id_iterator(map_.begin()); }
    id_iterator id_end() const { return id_iterator(map_.end()); }
    size_t id_size() const { return map_.size(); }
    void swap (IDVectorMap & other);
    IDVectorMap& operator=(IDVectorMap const& rhs);
  private:
    C collection_;
    map map_;
  };
  
  template <typename ID, typename C, typename P>
  inline
  void
  IDVectorMap<ID, C, P>::swap(IDVectorMap<ID, C, P> & other) {
    collection_.swap(other.collection_);
    map_.swap(other.map_);
  }

  template <typename ID, typename C, typename P>
  inline
  IDVectorMap<ID, C, P>&
  IDVectorMap<ID, C, P>::operator=(IDVectorMap<ID, C, P> const& rhs) {
    IDVectorMap<ID, C, P> temp(rhs);
    this->swap(temp);
    return *this;
  }

  // free swap function
  template <typename ID, typename C, typename P>
  inline
  void
  swap(IDVectorMap<ID, C, P> & a, IDVectorMap<ID, C, P> & b) {
    a.swap(b);
  }

}

#endif
