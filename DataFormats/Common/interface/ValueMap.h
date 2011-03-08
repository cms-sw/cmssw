#ifndef DataFormats_Common_ValueMap_h
#define DataFormats_Common_ValueMap_h
/* \class ValueMap<T>
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ValueMap.h,v 1.18 2010/09/01 19:48:30 chrjones Exp $
 *
 */

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <vector>
#include <map>
#include <iterator>
#include <algorithm>

namespace edm {
  namespace helper {
    template<typename Map>
    class Filler {
    private:
      typedef std::vector<size_t> index_vector;
      typedef std::vector<typename Map::value_type> value_vector;
      typedef std::map<ProductID, value_vector> value_map;
      typedef typename Map::offset offset;
      typedef typename Map::id_offset_vector id_offset_vector;
    public:
      explicit Filler(Map & map) : 
	map_(map) { 
	add(map);
      }
      void add(const Map & map) {
	if (map.empty()) return;
	typename id_offset_vector::const_iterator j = map.ids_.begin();
	const typename id_offset_vector::const_iterator end = map.ids_.end();
	size_t i = 0;
	const size_t size = map.values_.size();
	std::pair<ProductID, offset> id = *j;
	do {
	  ProductID id = j->first;
	  ++j;
	  size_t max = (j == end ? size : j->second);
	  typename value_map::iterator f = values_.find(id);
	  if(f!=values_.end()) throwAdd();
	  value_vector & values = values_.insert(std::make_pair(id, value_vector())).first->second;
	  while(i!=max)
	    values.push_back( map.values_[i++] );
	} while(j != end);
      }
      template<typename H, typename I>
      void insert(const H & h, I begin, I end) {
	ProductID id = h.id();
	size_t size = h->size(), sizeIt = end - begin;
	if(sizeIt!=size) throwFillSize();
	typename value_map::const_iterator f = values_.find(id);
	if(f != values_.end()) throwFillID(id);
	value_vector & values = values_.insert(make_pair(id, value_vector(size))).first->second;
	std::copy(begin, end, values.begin());
      }
      void fill() {
	map_.clear();
	offset off = 0;
	for(typename value_map::const_iterator i = values_.begin(); i != values_.end(); ++i) {
	  ProductID id = i->first;
	  map_.ids_.push_back(std::make_pair(id, off));
	  const value_vector & values = i->second;
	  for(typename value_vector::const_iterator j = values.begin(); j != values.end(); ++j) {
	    map_.values_.push_back( *j );
	    ++off;
	  }
	}
      }

    protected:
      Map & map_;

    private:
      value_map values_;
      void throwFillSize() const {
	Exception::throwThis(errors::InvalidReference,
	  "ValueMap::Filler: handle and reference "
	  "collections should the same size\n");	
      }
      void throwFillID(ProductID id) const {
	Exception e(errors::InvalidReference);
	e << "index map has already been filled for id: " << id << "\n";
	e.raise();
      }
      void throwAdd() const {
	Exception::throwThis(errors::InvalidReference,
	  "ValueMap: trying to add entries for an already existing product\n");
      }
    };
  }
  
  template<typename T>
  class ValueMap {
  public:
    typedef T value_type;
    typedef std::vector<value_type> container;
    typedef unsigned int offset;
    typedef std::vector<std::pair<ProductID, offset> > id_offset_vector;
    typedef typename container::reference       reference_type;
    typedef typename container::const_reference const_reference_type;

    ValueMap() { }

    void swap(ValueMap& other) {
      values_.swap(other.values_);
      ids_.swap(other.ids_);
    }

    ValueMap& operator=(ValueMap const& rhs) {
      ValueMap temp(rhs);
      this->swap(temp);
      return *this;
    }

    template<typename RefKey>
    const_reference_type operator[](const RefKey & r) const {
      return get(r.id(), r.key());
    }
    // raw index of a given (id,key) pair
    size_t rawIndexOf(ProductID id, size_t idx) const {
      typename id_offset_vector::const_iterator f = getIdOffset(id);
      if(f==ids_.end()) throwNotExisting();
      offset off = f->second;
      size_t j = off+idx;
      if(j >= values_.size()) throwIndexBound();
      return j;
    }
    const_reference_type get(ProductID id, size_t idx) const { 
      return values_[rawIndexOf(id,idx)];
    }
    template<typename RefKey>
    reference_type operator[](const RefKey & r) {
      return get(r.id(), r.key());
    }
    reference_type get(ProductID id, size_t idx) { 
      return values_[rawIndexOf(id,idx)];
    }

    ValueMap<T> & operator+=(const ValueMap<T> & o) {
      add(o);
      return *this;
    }
    bool contains(ProductID id) const {
      return getIdOffset(id) != ids_.end();
    }
    size_t size() const { return values_.size(); }
    size_t idSize() const { return ids_.size(); }
    bool empty() const { return values_.empty(); }
    void clear() { values_.clear(); ids_.clear(); }

    typedef helper::Filler<ValueMap<T> > Filler;

    struct const_iterator {
      typedef ptrdiff_t difference_type;
      const_iterator():values_(0) {}
      ProductID id() const { return i_->first; }
      typename container::const_iterator begin() const { 
	return values_->begin() + i_->second; 
      }
      typename container::const_iterator end() const { 
	if(i_ == end_) return values_->end();
	id_offset_vector::const_iterator end = i_; ++end;
	if(end == end_) return values_->end();
	return values_->begin() + end->second; 
      }
      size_t size() const { return end() - begin(); }
      const T & operator[](size_t i) { return *(begin()+i); }
      const_iterator& operator++() { ++i_; return *this; }
      const_iterator operator++(int) { const_iterator ci = *this; ++i_; return ci; }
      const_iterator& operator--() { --i_; return *this; }
      const_iterator operator--(int) { const_iterator ci = *this; --i_; return ci; }
      difference_type operator-(const const_iterator & o) const { return i_ - o.i_; }
      const_iterator operator+(difference_type n) const { return const_iterator(i_ + n, end_, values_); }
      const_iterator operator-(difference_type n) const { return const_iterator(i_ - n, end_, values_); }
      bool operator<(const const_iterator & o) const { return i_ < o.i_; }
      bool operator==(const const_iterator& ci) const { return i_ == ci.i_; }
      bool operator!=(const const_iterator& ci) const { return i_ != ci.i_; }
      const_iterator & operator +=(difference_type d) { i_ += d; return *this; }
      const_iterator & operator -=(difference_type d) { i_ -= d; return *this; }      
    private:
      const_iterator(const id_offset_vector::const_iterator & i_,
		     const id_offset_vector::const_iterator & end,
		     const container * values) :
	values_(values), i_(i_), end_(end) { }
      const container * values_;
      id_offset_vector::const_iterator i_, end_;
      friend class ValueMap<T>;
    };

    const_iterator begin() const { return const_iterator(ids_.begin(), ids_.end(), &values_); }
    const_iterator end() const { return const_iterator(ids_.end(), ids_.end(), &values_); }

    /// meant to be used in AssociativeIterator, not by the ordinary user
    const id_offset_vector & ids() const { return ids_; }
    /// meant to be used in AssociativeIterator, not by the ordinary user
    const_reference_type get(size_t idx) const { return values_[idx]; }
    
    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  protected:
    container values_;
    id_offset_vector ids_;

    typename id_offset_vector::const_iterator getIdOffset(ProductID id) const {
      typename id_offset_vector::const_iterator i = std::lower_bound(ids_.begin(), ids_.end(), id, IDComparator());
      if(i==ids_.end()) return i;
      return i->first == id ? i : ids_.end();
    }

    void throwIndexBound() const {
      Exception::throwThis(errors::InvalidReference, "ValueMap: index out of upper boundary\n");
    }

  private:
    struct IDComparator {
      bool operator()(const std::pair<ProductID, offset> & p, const ProductID & id)  {
	return p.first < id;
      }
    };
    void throwNotExisting() const {
      Exception::throwThis(errors::InvalidReference, "ValueMap: no associated value for given product and index\n");
    }

    void add( const ValueMap<T> & o ) {
      Filler filler(*this);
      filler.add(o);
      filler.fill();
    }

    friend class helper::Filler<ValueMap<T> >;
  }; 

  template<typename T>
  inline ValueMap<T> operator+( const ValueMap<T> & a1,
				    const ValueMap<T> & a2 ) {
    ValueMap<T> a = a1;
    a += a2;
    return a;
  }

  // Free swap function
  template <typename T>
  inline
  void swap(ValueMap<T>& lhs, ValueMap<T>& rhs) {
    lhs.swap(rhs);
  }

}
#endif
