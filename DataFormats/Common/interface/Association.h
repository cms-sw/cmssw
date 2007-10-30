#ifndef DataFormats_Common_Association_h
#define DataFormats_Common_Association_h
/* \class Association<RefProd>
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Association.h,v 1.1 2007/10/29 12:53:35 llista Exp $
 *
 */

#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/ValueMap.h"

namespace edm {
  
  template<typename C>
  class Association : private ValueMap<int> {
  public:
    typedef int index; // negative index == null reference
    typedef ValueMap<index> base;
    typedef typename base::offset offset;
    typedef edm::RefProd<C> RefProd; // could be specialized for View
    typedef Ref<typename RefProd::product_type> RefVal;

    Association() : base() { }
    template<typename H>
    explicit Association(const H & h) : base(), ref_(h) { }
    
    template<typename RefKey>
      RefVal operator[](const RefKey & r) const {
      return get(r.id(), r.key());
    }
    RefVal get(ProductID id, size_t idx) const { 
      index i = base::get(id, idx);
      if(i < 0) return RefVal(); 
      size_t k = i;
      if (k >= ref_->size()) throwIndexMapBound();
      return RefVal(ref_,k);
    }
    Association<C> & operator+=(const Association<C> & o) {
      add(o);
      return *this;
    }
    void setRef(const RefProd & ref) {
      if(ref_.isNull() ) {
	ref_ = ref;
      } else {
	if(ref_.id() != ref.id()) throwRefSet();
      }
    }
    bool contains(ProductID id) const { return base::contains(id); }
    size_t size() const { return base::size(); }
    bool empty() const { return base::empty(); }
    void clear() { base::clear(); }

    class Filler : public base::Filler {
    public:
      explicit Filler(Association<C> & association) : 
	base::Filler(association) { }
    };

  private:
    RefProd ref_;
    void throwIndexMapBound() const {
      throw Exception(errors::InvalidReference)
	<< "Association: index in the map out of upper boundary";
    }
    void throwRefSet() const {
      throw Exception(errors::InvalidReference) 
	<< "Association: reference to product already set";
    }
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
