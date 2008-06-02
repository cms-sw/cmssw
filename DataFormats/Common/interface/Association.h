#ifndef DataFormats_Common_Association_h
#define DataFormats_Common_Association_h
/* \class Association<RefProd>
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Association.h,v 1.5 2007/11/07 10:25:13 llista Exp $
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
    typedef edm::RefProd<C> refprod_type; // could be specialized for View
    typedef Ref<typename refprod_type::product_type> reference_type;

    Association() : base() { }
    template<typename H>
    explicit Association(const H & h) : base(), ref_(h) { }
    
    template<typename RefKey>
      reference_type operator[](const RefKey & r) const {
      return get(r.id(), r.key());
    }
    reference_type get(ProductID id, size_t idx) const { 
      typename id_offset_vector::const_iterator f = getIdOffset(id);
      if(f==ids_.end()||f->first != id) return reference_type();
      offset off = f->second;
      size_t j = off+idx;
      if(j >= values_.size()) throwIndexBound();
      index i = values_[j];
      if(i < 0) return reference_type(); 
      size_t k = i;
      if (k >= ref_->size()) throwIndexMapBound();
      return reference_type(ref_,k);
    }
    Association<C> & operator+=(const Association<C> & o) {
      add(o);
      return *this;
    }
    void setRef(const refprod_type & ref) {
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
    refprod_type ref() const { return ref_; }
    void swap(Association& other) {
      this->ValueMap<int>::swap(other);
      ref_.swap(other.ref_);
    }
    Association& operator=(Association const& rhs) {
      Association temp(rhs);
      this->swap(temp);
      return *this;
    }

    class Filler : public helper::Filler<Association<C> > {
      typedef helper::Filler<Association<C> > base;
    public:
      explicit Filler(Association<C> & association) : 
	base(association) { }
      void add(const Association<C> & association) {
	base::map_.setRef(association.ref());
	base::add(association);
      }
    };

  private:
    refprod_type ref_;
    void throwIndexMapBound() const {
      throw Exception(errors::InvalidReference)
	<< "Association: index in the map out of upper boundary\n";
    }
    void throwRefSet() const {
      throw Exception(errors::InvalidReference) 
	<< "Association: reference to product already set\n";
    }

    void add( const Association<C> & o ) {
      Filler filler(*this);
      filler.add(o);
      filler.fill();
    }

    friend class helper::Filler<Association<C> >;
  }; 
  
  // Free swap function
  template <typename C>
  inline void swap(Association<C>& lhs, Association<C>& rhs) {
    lhs.swap(rhs);
  }

  template<typename C>
  inline  Association<C> operator+( const Association<C> & a1,
				       const Association<C> & a2 ) {
    Association<C> a = a1;
    a += a2;
    return a;
  }
}

#endif
