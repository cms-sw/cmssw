#ifndef DataFormats_Common_Association_h
#define DataFormats_Common_Association_h
/* \class Association<RefProd>
 *
 * \author Luca Lista, INFN
 *
 * \version $Id: Association.h,v 1.8 2009/11/04 15:30:19 wmtan Exp $
 *
 */

#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
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

    // import this function from ValueMap<int>
    using base::rawIndexOf;
 
    template<typename RefKey>
      reference_type operator[](const RefKey & r) const {
      return get(r.id(), r.key());
    }

    /// meant to be used internally or in AssociativeIterator, not by the ordinary user
    reference_type get(size_t rawIdx) const { 
      index i = values_[rawIdx];
      if(i < 0) return reference_type(); 
      size_t k = i;
      if (k >= ref_->size()) throwIndexMapBound();
      return reference_type(ref_,k);
    }

    reference_type get(ProductID id, size_t idx) const { 
      return get(rawIndexOf(id,idx));
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

    /// meant to be used in AssociativeIterator, not by the ordinary user
    const id_offset_vector & ids() const { return ids_; }
    /// meant to be used in AssociativeIterator, not by the ordinary user
    using base::id_offset_vector;
    
    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    refprod_type ref_;
    void throwIndexMapBound() const {
      Exception::throwThis(errors::InvalidReference, "Association: index in the map out of upper boundary\n");
    }
    void throwRefSet() const {
      Exception::throwThis(errors::InvalidReference, "Association: reference to product already set\n");
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
