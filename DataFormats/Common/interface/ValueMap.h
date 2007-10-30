#ifndef DataFormats_Common_ValueMap_h
#define DataFormats_Common_ValueMap_h
/* \class ValueMap<T>
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: ValueMap.h,v 1.1 2007/10/29 12:53:35 llista Exp $
 *
 */

#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <vector>
#include <map>
#include <iterator>

namespace edm {
  
  template<typename T>
  class ValueMap {
  public:
    typedef T value_type;
    typedef std::vector<value_type> container;
    typedef unsigned int offset;
    typedef std::vector<std::pair<ProductID, offset> > id_offset_vector;

    ValueMap() { }
    template<typename RefKey>
    value_type operator[](const RefKey & r) const {
      return get(r.id(), r.key());
    }
    value_type get(ProductID id, size_t idx) const { 
      id_offset_vector::const_iterator f =
	std::lower_bound(ids_.begin(), ids_.end(), id, IDComparator());
      if(f==ids_.end()||f->first != id) throwNotExisting();
      offset off = f->second;
      size_t j = off+idx;
      if(j >= container_.size()) throwIndexBound();
      return container_[j];
    }
    ValueMap<T> & operator+=(const ValueMap<T> & o) {
      add(o);
      return *this;
    }
    bool contains(ProductID id) const {
      return std::lower_bound(ids_.begin(), ids_.end(), id, IDComparator()) != ids_.end();
    }
    size_t size() const { return container_.size(); }
    bool empty() const { return container_.empty(); }
    void clear() { container_.clear(); ids_.clear(); }

    class Filler {
      typedef std::vector<size_t> index_vector;
      typedef std::vector<value_type> value_vector;
      typedef std::map<ProductID, value_vector> value_map;
    public:
      explicit Filler(ValueMap<T> & association) : 
	association_(association) { 
	add(association);
      }
      void add(const ValueMap<T> & association) {
	if (association.empty()) return;
	id_offset_vector::const_iterator j = association.ids_.begin();
	const id_offset_vector::const_iterator end = association.ids_.end();
	size_t i = 0;
	const size_t size = association.container_.size();
	std::pair<ProductID, offset> id = *j;
	do {
	  ProductID id = j->first;
	  ++j;
	  size_t max = (j == end ? size : j->second);
	  typename value_map::iterator f = values_.find(id);
	  if(f!=values_.end())
	    throw Exception(errors::InvalidReference)
	      << "ValueMap: trying to add entries for an already existing product";
	  value_vector & values = values_.insert(std::make_pair(id, value_vector())).first->second;
	  while(i!=max)
	    values.push_back( association.container_[i++] );
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
	association_.clear();
	offset off = 0;
	for(typename value_map::const_iterator i = values_.begin(); i != values_.end(); ++i) {
	  ProductID id = i->first;
	  association_.ids_.push_back(std::make_pair(id, off));
	  const value_vector & values = i->second;
	  for(typename value_vector::const_iterator j = values.begin(); j != values.end(); ++j) {
	    association_.container_.push_back( *j );
	    ++off;
	  }
	}
      }
    private:
      ValueMap<T> & association_;
      value_map values_;
      void throwFillSize() const {
	throw Exception(errors::InvalidReference)
	  << "ValueMap::Filler: handle and reference"
	  << "collections should the same size";	
      }
      void throwFillID(ProductID id) const {
	throw Exception(errors::InvalidReference)
	  << "index map has already been filled for id: " << id;
      }
    };

  protected:
    void add( const ValueMap<T> & o ) {
      Filler filler(*this);
      filler.add(o);
      filler.fill();
    }

  private:
    struct IDComparator {
      bool operator()(const std::pair<ProductID, offset> & p, const ProductID & id)  {
	return p.first < id;
      }
    };
    void throwIndexBound() const {
      throw Exception(errors::InvalidReference)
	<< "ValueMap: index out of upper boundary";
    }
    void throwNotExisting() const {
      throw Exception(errors::InvalidReference)
	<< "ValueMap: no associated value for given product and index";
    }

    container container_;
    id_offset_vector ids_;
    friend class Filler;
  }; 
  
}

template<typename T>
inline  edm::ValueMap<T> operator+( const edm::ValueMap<T> & a1,
				    const edm::ValueMap<T> & a2 ) {
  edm::ValueMap<T> a = a1;
  a += a2;
  return a;
}
#endif
