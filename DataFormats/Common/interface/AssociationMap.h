#ifndef Common_AssociationMap_h
#define Common_AssociationMap_h
/** \class edm::AssociationMap
 *
 * one-to-many or one-to-one associative map using EDM references
 *
 * \author Luca Lista, INFN
 *
 * $Id: AssociationMap.h,v 1.20 2006/08/28 07:14:24 llista Exp $
 *
 */
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <map>
#include <vector>

namespace edm {
  namespace helpers {
    template<typename K, typename V>
    struct KeyVal {
      typedef K key_type;
      typedef V value_type;
      KeyVal() { }
      KeyVal( const K & k, const V & v ) : key( k ), val( v ) { }
      K key;
      V val;
    };

    template<typename K>
    struct Key {
      typedef K key_type;
      Key() { }
      Key( const K & k ) : key( k ) { }
      K key;
    };

    /// throw if r hasn't the same id as rp
    template<typename RP, typename R>
    void checkRef( const RP & rp, const R & r ) {
      if ( rp.id() != r.id() )
	throw edm::Exception( edm::errors::InvalidReference, "invalid reference" );
    }
  }

  template<typename CKey, typename CVal, typename index = unsigned int>
  class OneToOne {
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// reference to "value" collection
    typedef edm::RefProd<CVal> ValRefProd;
    /// values reference collection type
    typedef edm::Ref<CVal> val_type;
  public:
    /// insert key type
    typedef edm::Ref<CKey> key_type;
    /// insert val type
    typedef edm::Ref<CVal> data_type;
    /// index type
    typedef index index_type;
    /// map type
    typedef std::map<index_type, index_type> map_type;
    /// value type
    typedef helpers::KeyVal<key_type, val_type> value_type;
    /// reference set type
    typedef helpers::KeyVal<KeyRefProd, ValRefProd> ref_type;
    /// insert in the map
    static void insert(  ref_type & ref, map_type & m,
			const key_type & k, const data_type & v ) {
     if ( k.isNull() || v.isNull() )
	throw edm::Exception( edm::errors::InvalidReference )
	  << "can't insert null references in AssociationMap";
      if ( ref.key.isNull() ) {
	ref.key = KeyRefProd( k );
	ref.val = ValRefProd( v );
      }
      helpers::checkRef( ref.key, k ); helpers::checkRef( ref.val, v );
      index_type ik = index_type( k.key() ), iv = index_type( v.key() );
      m[ ik ] = iv;
    }
    /// return values collection
    static val_type val( const ref_type & ref, index_type iv ) {
      return val_type( ref.val, iv );
    }
    /// size of data_type
    static typename map_type::size_type size( const index_type & v ) { return 1; }
  };

  template<typename CKey, typename CVal, typename index = unsigned int>
  class OneToMany {
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// reference to "value" collection
    typedef edm::RefProd<CVal> ValRefProd;
    /// values reference collection type
    typedef edm::RefVector<CVal> val_type;
  public:
    /// insert key type
    typedef edm::Ref<CKey> key_type;
    /// insert val type
    typedef edm::Ref<CVal> data_type;
    /// index type
    typedef index index_type;
    /// map type
    typedef std::map<index_type, std::vector<index_type> > map_type;
    /// value type
    typedef helpers::KeyVal<key_type, val_type> value_type;
    /// reference set type
    typedef helpers::KeyVal<KeyRefProd, ValRefProd> ref_type;
    /// insert in the map
    static void insert( ref_type & ref, map_type & m,
			const key_type & k, const data_type & v ) {
     if ( k.isNull() || v.isNull() )
	throw edm::Exception( edm::errors::InvalidReference )
	  << "can't insert null references in AssociationMap";
      if ( ref.key.isNull() ) {
	ref.key = KeyRefProd( k );
	ref.val = ValRefProd( v );
      }
      helpers::checkRef( ref.key, k ); helpers::checkRef( ref.val, v );
      index_type ik = index_type( k.key() ), iv = index_type( v.key() );
      m[ ik ].push_back( iv );
    }
    /// return values collection
    static val_type val( const ref_type & ref, const std::vector<index_type> & iv ) {
      val_type v( ref.val.id() );
      for( typename std::vector<index_type>::const_iterator idx = iv.begin(); idx != iv.end(); ++ idx )
	v.push_back( edm::Ref<CVal>( ref.val, * idx ) );
      return v;
    }
    /// size of data_type
    static typename map_type::size_type size( const std::vector<index_type> & v ) { return v.size(); }
  };

  template<typename CKey, typename Val, typename index = unsigned int>
  class OneToValue {
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// values reference collection type
    typedef Val val_type;
  public:
    /// insert key type
    typedef edm::Ref<CKey> key_type;
    /// insert val type
    typedef Val data_type;
    /// index type
    typedef index index_type;
    /// map type
    typedef std::map<index_type, Val> map_type;
    /// value type
    typedef helpers::KeyVal<key_type, val_type> value_type;
    /// reference set type
    typedef helpers::Key<KeyRefProd> ref_type;
    /// insert in the map
    static void insert( ref_type & ref, map_type & m,
			const key_type & k, const data_type & v ) {
     if ( k.isNull() )
	throw edm::Exception( edm::errors::InvalidReference )
	  << "can't insert null references in AssociationMap";
      if ( ref.key.isNull() ) {
	ref.key = KeyRefProd( k );
      }
      helpers::checkRef( ref.key, k );
      index_type ik = index_type( k.key() );
      m[ ik ] = v;
    }
    /// return values collection
    static val_type val( const ref_type & ref, const Val & v ) {
      return v;
    }
    /// size of data_type
    static typename map_type::size_type size( const data_type & ) { return 1; }
  };

  template<typename Tag>
  struct AssociationMap {
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
    typedef typename Tag::value_type value_type;
    /// result type
    typedef typename value_type::value_type result_type;
    /// transient map type
    typedef typename std::map<index_type, value_type> transient_map_type;

    /// const iterator
    struct const_iterator {
      typedef self::value_type value_type;
      typedef ptrdiff_t difference_type;
      typedef value_type * pointer;
      typedef value_type & reference;
      typedef typename map_type::const_iterator::iterator_category iterator_category;
      const_iterator() { }
      const_iterator( const self * map, typename map_type::const_iterator mi ) :
	map_( map ), i( mi ) { }
      const_iterator & operator=( const const_iterator & it ) {
	map_ = it.map_; i = it.i; return *this;
      }
      const_iterator& operator++() { ++i; return *this; }
      const_iterator operator++( int ) { const_iterator ci = *this; ++i; return ci; }
      const_iterator& operator--() { --i; return *this; }
      const_iterator operator--( int ) { const_iterator ci = *this; --i; return ci; }
      bool operator==( const const_iterator& ci ) const { return i == ci.i; }
      bool operator!=( const const_iterator& ci ) const { return i != ci.i; }
      const value_type & operator *() const { return (*map_)[ i->first ]; }
      const value_type * operator->() const { return & operator *(); }
    private:
      const self * map_;
      typename map_type::const_iterator i;
    };

    /// default constructor
    AssociationMap() { }
    /// clear map
    void clear() { map_.clear(); transientMap_.clear(); }
    /// map size
    size_type size() const { return map_.size(); }
    /// return true if empty
    bool empty() const { return map_.empty(); }
    /// insert an association
    void insert( const key_type & k, const data_type & v ) {
      Tag::insert( ref_, map_, k, v );
    }
    /// first iterator over the map (read only)
    const_iterator begin() const { return const_iterator( this, map_.begin() );  }
    /// last iterator over the map (read only)
    const_iterator end() const { return const_iterator( this, map_.end() );  }
    /// find element with specified reference key
    const_iterator find( const key_type & k ) const {
      if ( ref_.key.id() != k.id() ) return end();
      return find( k.key() );
    }
    /// erase the element whose key is k
    size_type erase( const key_type& k ) {
      index_type i = k.key();
      transientMap_.erase( i );
      return map_.erase( i );
    }
    /// find element with specified reference key
    const result_type & operator[]( const key_type & k ) const {
      helpers::checkRef( ref_.key, k );
      return operator[]( k.key() ).val;
    }
    /// number of associations to a key
    size_type numberOfAssociations( const key_type & k ) const {
      if ( ref_.key.id() != k.id() ) return 0;
      typename map_type::const_iterator f = map_.find( k.key() );
      if ( f == map_.end() ) return 0;
      return Tag::size( f->second );
    }

  private:
    /// find helper
    struct Find :
      public std::binary_function<const self&, size_type, const value_type *> {
      typedef Find self;
      typename self::result_type operator()( typename self::first_argument_type c,
					     typename self::second_argument_type i ) {
	return & ( * c.find( i ) );
      }
    };
    /// reference set
    ref_type ref_;
    /// index map
    map_type map_;
    /// transient reference map
    mutable transient_map_type transientMap_;
    /// find element with index i
    const_iterator find( size_type i ) const {
      typename map_type::const_iterator f = map_.find( i );
      if ( f == map_.end() ) return end();
      return const_iterator( this, f );
    }
    /// return value_typeelement with key i
    const value_type & operator[]( size_type i ) const {
      typename transient_map_type::const_iterator tf = transientMap_.find( i );
      if ( tf == transientMap_.end() ) {
	typename map_type::const_iterator f = map_.find( i );
	if ( f == map_.end() )
	  throw edm::Exception( edm::errors::InvalidReference )
	    << "can't find reference in AssociationMap at position " << i;
	value_type v( key_type( ref_.key, i ), Tag::val( ref_, f->second ) );
	std::pair<typename transient_map_type::const_iterator, bool> ins =
	  transientMap_.insert( std::make_pair( i, v ) );
	return ins.first->second;
      } else {
	return tf->second;
      }
    }
    friend struct const_iterator;
    friend struct Find;
    friend struct refhelper::FindTrait<self,value_type>;
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
