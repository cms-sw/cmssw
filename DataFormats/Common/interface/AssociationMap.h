#ifndef DataFormats_Common_AssociationMap_h
#define DataFormats_Common_AssociationMap_h
/** \class edm::AssociationMap
 *
 * one-to-many or one-to-one associative map using EDM references
 *
 * \author Luca Lista, INFN
 *
 * $Id: AssociationMap.h,v 1.40 2010/09/01 19:48:30 chrjones Exp $
 *
 */
#include "DataFormats/Common/interface/CMS_CLASS_VERSION.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/OneToValue.h"
#include "DataFormats/Common/interface/OneToOne.h"
#include "DataFormats/Common/interface/OneToMany.h"
#include "DataFormats/Common/interface/OneToManyWithQuality.h"

namespace edm {
  template<typename Tag>
  class AssociationMap {
    /// insert key type
    typedef typename Tag::val_type internal_val_type;
  public:
    /// self type
    typedef AssociationMap<Tag> self;
    /// index type
    typedef typename Tag::index_type index_type;
    /// insert key type
    typedef typename Tag::key_type key_type;
    /// insert data type
    typedef typename Tag::data_type data_type;
    /// reference set type
    typedef typename Tag::ref_type ref_type;
    /// map type
    typedef typename Tag::map_type map_type;
    /// size type
    typedef typename map_type::size_type size_type;
    /// value type
    typedef helpers::KeyVal<key_type, internal_val_type> value_type;
    /// result type
    typedef typename value_type::value_type result_type;
    /// transient map type
    typedef typename std::map<index_type, value_type> internal_transient_map_type;

    /// const iterator
    struct const_iterator {
      typedef typename self::value_type value_type;
      typedef ptrdiff_t difference_type;
      typedef value_type * pointer;
      typedef value_type & reference;
      typedef typename map_type::const_iterator::iterator_category iterator_category;
      const_iterator(): map_(0) { }
      const_iterator(const self * map, typename map_type::const_iterator mi) :
	map_(map), i(mi) { }
      const_iterator & operator=(const const_iterator & it) {
	map_ = it.map_; i = it.i; return *this;
      }
      const_iterator& operator++() { ++i; return *this; }
      const_iterator operator++(int) { const_iterator ci = *this; ++i; return ci; }
      const_iterator& operator--() { --i; return *this; }
      const_iterator operator--(int) { const_iterator ci = *this; --i; return ci; }
      bool operator==(const const_iterator& ci) const { return i == ci.i; }
      bool operator!=(const const_iterator& ci) const { return i != ci.i; }
      const value_type & operator *() const { return (*map_)[ i->first ]; }
      const value_type * operator->() const { return &operator *(); }
    private:
      const self * map_;
      typename map_type::const_iterator i;
    };

    /// default constructor
    AssociationMap() { }
    /// default constructor
    explicit
    AssociationMap(const ref_type & ref) : ref_(ref) { }
    /// clear map
    void clear() { map_.clear(); transientMap_.clear(); }
    /// map size
    size_type size() const { return map_.size(); }
    /// return true if empty
    bool empty() const { return map_.empty(); }
    /// insert an association
    void insert(const key_type & k, const data_type & v) {
      Tag::insert(ref_, map_, k, v);
    }
    void insert(const value_type & kv) {
      Tag::insert(ref_, map_, kv.key, kv.val);
    }
    /// first iterator over the map (read only)
    const_iterator begin() const { return const_iterator(this, map_.begin());  }
    /// last iterator over the map (read only)
    const_iterator end() const { return const_iterator(this, map_.end());  }
    /// find element with specified reference key
    const_iterator find(const key_type & k) const {
      if (ref_.key.id() != k.id()) return end();
      return find(k.key());
    }
    /// erase the element whose key is k
    size_type erase(const key_type& k) {
      index_type i = k.key();
      transientMap_.erase(i);
      return map_.erase(i);
    }
    /// find element with specified reference key
    const result_type & operator[](const key_type & k) const {
      helpers::checkRef(ref_.key, k);
      return operator[](k.key()).val;
    }
    /// number of associations to a key
    size_type numberOfAssociations(const key_type & k) const {
      if (ref_.key.id() != k.id()) return 0;
      typename map_type::const_iterator f = map_.find(k.key());
      if (f == map_.end()) return 0;
      return Tag::size(f->second);
    }
    /// return ref-prod structure
    const ref_type & refProd() const { return ref_; }

    /// fill and return a transient map
    /// required for ROOT interactive usage
    typename Tag::transient_map_type map() { 
      return Tag::transientMap( ref_, map_ ); 
    }
    /// fill and return a transient key vector
    /// required for ROOT interactive usage
    typename Tag::transient_key_vector keys() { 
      return Tag::transientKeyVector( ref_, map_ ); 
    }
    /// fill and return a transient key vector
    /// required for ROOT interactive usage
    typename Tag::transient_val_vector values() { 
      return Tag::transientValVector( ref_, map_ ); 
    }
     /// post insert action
    void post_insert() { Tag::sort(map_); }

    // Find should be private!  However, generated reflex dictionaries do not compile
    //  with gcc 3.4.5 if Find is private.  Reflex should fix this!!
    /// find helper
    struct Find :
      public std::binary_function<const self&, size_type, const value_type *> {
      typedef Find self;
      const value_type * operator()(typename self::first_argument_type c,
				     typename self::second_argument_type i) {
	return &(*c.find(i));
      }
    };

    //Used by ROOT storage
    CMS_CLASS_VERSION(10)

  private:
    /// reference set
    ref_type ref_;
    /// index map
    map_type map_;
    /// transient reference map
    mutable internal_transient_map_type transientMap_;
    /// find element with index i
    const_iterator find(size_type i) const {
      typename map_type::const_iterator f = map_.find(i);
      if (f == map_.end()) return end();
      return const_iterator(this, f);
    }
    /// return value_typeelement with key i
    const value_type & operator[](size_type i) const {
      typename internal_transient_map_type::const_iterator tf = transientMap_.find(i);
      if (tf == transientMap_.end()) {
	typename map_type::const_iterator f = map_.find(i);
	if (f == map_.end())
	  Exception::throwThis(edm::errors::InvalidReference, "can't find reference in AssociationMap at position ", i);
	value_type v(key_type(ref_.key, i), Tag::val(ref_, f->second));
	std::pair<typename internal_transient_map_type::const_iterator, bool> ins =
	  transientMap_.insert(std::make_pair(i, v));
	return ins.first->second;
      } else {
	return tf->second;
      }
    }
    friend struct const_iterator;
    friend struct Find;
    friend struct refhelper::FindTrait<self,value_type>;
    template<typename, typename, typename> friend class OneToValue;
    template<typename, typename, typename> friend class OneToOne;
    template<typename, typename, typename> friend class OneToMany;
    template<typename, typename, typename, typename> friend class OneToManyWithQuality;
  };

  namespace refhelper {
    template<typename Tag>
    struct FindTrait<AssociationMap<Tag>,
		     typename AssociationMap<Tag>::value_type> {
      typedef typename AssociationMap<Tag>::Find value;
    };
  }

}

#endif
