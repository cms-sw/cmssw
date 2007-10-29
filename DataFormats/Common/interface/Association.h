#ifndef DataFormats_Common_Association_h
#define DataFormats_Common_Association_h
/* \class Association<RefProd>
 *
 * \author Luca Lista, INFN
 *
 * \version $Id$
 *
 */

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Provenance/interface/ProductID.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <vector>
#include <map>
#include <iterator>

namespace edm {
  
  template<typename C>
  class Association {
  public:
    typedef int index; // negative index == null reference
    typedef unsigned int offset;
    typedef edm::RefProd<C> RefProd; // could be specialized for View
    Association() { }
    template<typename H>
    explicit Association(const H & h) : ref_(h) { }
    typedef Ref<typename RefProd::product_type> RefVal;
    
    template<typename RefKey>
      RefVal operator[](const RefKey & r) const {
      return get(r.id(), r.key());
    }
    RefVal get(ProductID id, size_t idx) const { 
      std::vector<std::pair<ProductID, offset> >::const_iterator f =
	std::lower_bound(ids_.begin(), ids_.end(), id, IDComparator());
      if(f==ids_.end()||f->first != id) return RefVal();
      offset off = f->second;
      size_t j = off+idx;
      if(j >= indices_.size()) throwIndexBound();
      index i = indices_[j];
      if(i < 0) return RefVal(); 
      size_t k = i;
      if (k >= ref_->size()) throwIndexMapBound();
      return RefVal(ref_,k);
    }
    bool empty() const { return indices_.empty(); }
    void clear() { indices_.clear(); ids_.clear(); }
    Association<C> & operator+=( const Association<C> & o ) {
      Filler filler(*this);
      filler.add(o);
      filler.fill();
      return *this;
    }
    size_t size() const { return indices_.size(); }
  private:
    struct IDComparator {
      bool operator()(const std::pair<ProductID, offset> & p, const ProductID & id)  {
	return p.first < id;
      }
    };
    void throwIndexBound() const {
      throw Exception(errors::InvalidReference)
	<< "Association: index out of upper boundary";
    }
    void throwIndexMapBound() const {
      throw Exception(errors::InvalidReference)
	<< "Association: index in the map out of upper boundary";
     }
  public:
    
    class Filler {
    public:
      explicit Filler(Association<C> & container) : 
	container_(container) { 
	add(container);
      }
      void add( const Association<C> & container) {
	if ( container.empty() ) return;
	std::vector<std::pair<ProductID, offset> >::const_iterator j = container.ids_.begin();
	const std::vector<std::pair<ProductID, offset> >::const_iterator end = container.ids_.end();
	size_t i = 0;
	const size_t size = container.indices_.size();
	std::pair<ProductID, offset> id = *j;
	do {
	  ProductID id = j->first;
	  ++j;
	  size_t max = ( j == end ? size : j->second );
	  std::map<ProductID, std::vector<index> >::iterator f = indices_.find(id);
	  if(f!=indices_.end())
	    throw Exception(errors::InvalidReference)
	      << "Association: trying to add entries for an already existing product";
	  std::vector<index> & indices = indices_.insert(std::make_pair(id, std::vector<index>())).first->second;
	  while( i!=max )
	    indices.push_back( container.indices_[i++] );
	} while( j != end );
      }
      template<typename H, typename I>
      void insert(const H & h, I begin, I end) {
	ProductID id = h.id();
	size_t size = h->size(), sizeIt = end-begin;
	if(sizeIt!=size) throwFillSize();
	std::map<ProductID, std::vector<index> >::const_iterator f = indices_.find(id);
	if(f != indices_.end()) throwFillID(id);
	std::vector<index> & idx = indices_.insert(make_pair(id,std::vector<index>(size))).first->second;
	std::copy(begin, end, idx.begin());
      }
      void setRef(const RefProd & ref) {
	if(container_.ref_.isNull() ) {
	  container_.ref_ = ref;
	} else {
	  if(container_.ref_.id() != ref.id()) throwRefSet();
	}
      }
      void fill() {
	container_.clear();
	offset off = 0;
	for(std::map<ProductID, std::vector<index> >::const_iterator i = indices_.begin();
	     i != indices_.end(); ++i) {
	  ProductID id = i->first;
	  container_.ids_.push_back(std::make_pair(id,off));
	  const std::vector<index> & indices = i->second;
	  for( std::vector<index>::const_iterator j = indices.begin(); j != indices.end(); ++j ) {
	    container_.indices_.push_back( *j );
	    ++off;
	  }
	}
      }
    private:
      Association<C> & container_;
      std::map<ProductID, std::vector<index> > indices_;
      void throwFillSize() const {
	throw Exception(errors::InvalidReference)
	  << "Association::Filler: handle and reference"
	  << "collections should the same size";	
      }
      void throwFillID(ProductID id) const {
	throw Exception(errors::InvalidReference)
	  << "index map has already been filled for id: " << id;
      }
      void throwRefSet() const {
	throw Exception(errors::InvalidReference) 
	  << "Association: reference to product already set";
      }
    };

  private:
    RefProd ref_;
    std::vector<index> indices_;
    std::vector<std::pair<ProductID, offset> > ids_;
    friend class Filler;
  }; 
  
}

template<typename C>
inline  edm::Association<C> operator+( const edm::Association<C> & a1,
				       const edm::Association<C> & a2 ) {
  edm::Association<C> a = a1;
  a += a2;
  return a;
}

#endif
