#ifndef Common_AssociationMap_h
#define Common_AssociationMap_h
/** \class edm::AssociationMap
 *
 * one-to-many reference map using EDM references
 * 
 * \author Luca Lista, INFN
 *
 * $Id: AssociationMap.h,v 1.11 2006/04/20 13:06:32 llista Exp $
 *
 */
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <map>
#include <vector>

namespace edm {

  template<typename K, typename V>
  struct KeyVal {
    KeyVal() { }
    KeyVal( const K & k, const V & v ) : key( k ), val( v ) { }
    K key;
    V val;
  };

  template<typename CVal, typename index>
  struct OneToOne {
    /// map type
    typedef std::map<index, index> map_type;
    /// index collection type
    typedef index idx_val;
    /// values reference collection type
    typedef edm::Ref<CVal> val_type;
    /// insert an index in the map
    static void insert( map_type & m, index ik, index iv ) {
      m[ ik ] = iv;
    }
    /// return values collection
    static val_type val( const edm::RefProd<CVal> & vp, idx_val iv ) {
      return val_type( vp, iv );
    }
  };

  template<typename CVal, typename index>
  struct OneToMany {
    /// map type    
    typedef std::map<index, std::vector<index> > map_type;
    /// index collection type
    typedef std::vector<index> idx_val;
    /// values reference collection type
    typedef edm::RefVector<CVal> val_type;
    /// insert an index in the map
    static void insert( map_type & m, index ik, index iv ) {
      m[ ik ].push_back( iv );
    }
    /// return values collection
    static val_type val( const edm::RefProd<CVal> & vp, const idx_val & iv ) {
      val_type v( vp.id() );
      for( typename std::vector<index>::const_iterator idx = iv.begin(); idx != iv.end(); ++ idx )
	v.push_back( edm::Ref<CVal>( vp, * idx ) );
      return v;
    }
  };

  template<typename CKey, typename CVal, 
	   template<typename, typename> class TagT, 
	   typename index = unsigned long>
  struct AssociationMap {
    /// tag type (OneToMany or OneToOne)
    typedef TagT<CVal, index> Tag;
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// reference to an object in "key" collection
    typedef edm::Ref<CKey> KeyRef;
    /// reference to "value" collection
    typedef edm::RefProd<CVal> ValRefProd;
    /// reference to an object in "value" collection
    typedef edm::Ref<CVal> ValRef;
    /// map type
    typedef typename Tag::map_type map_type;
    /// size type
    typedef typename map_type::size_type size_type;
    /// key/val pair
    typedef KeyVal<edm::Ref<CKey>, typename Tag::val_type> KeyVal;
    /// value type
    typedef KeyVal value_type;

    /// default constructor
    AssociationMap() { }
    /// constructor from product references
    AssociationMap( const KeyRefProd & k, const ValRefProd & v ) :
      keyRef_( k ), valRef_( v ) {
    }
    /// map size
    size_type size() const { return map_.size(); }
    /// return true if empty
    bool empty() const { return map_.empty(); }
    /// insert an association
    void insert( const KeyRef & k, const ValRef & v ) {
      if ( k.isNull() || v.isNull() )
	throw edm::Exception( edm::errors::InvalidReference )
	  << "can't insert null references in OneToOneAssociation";
      if ( keyRef_.isNull() ) {
	keyRef_ = KeyRefProd( k ); 
	valRef_ = ValRefProd( v );
      }
      checkKey( k ); checkVal( v );
      index ik = index( k.index() ), iv = index( v.index() );
      Tag::insert( map_, ik, iv );
    }
    /// const iterator
    struct const_iterator {
      typedef ptrdiff_t difference_type;
      typedef typename map_type::const_iterator::iterator_category iterator_category;
      const_iterator() { }
      const_iterator( const KeyRefProd & keyRef, const ValRefProd & valRef,
		      typename map_type::const_iterator mi ) : 
	keyRef_( keyRef ), valRef_( valRef ), i( mi ) { }
      const_iterator & operator=( const const_iterator & it ) { 
	keyRef_ = it.keyRef_; valRef_ = it.valRef_;
	i = it.i; 
	return *this; 
      }
      const_iterator& operator++() { ++i; return *this; }
      const_iterator operator++( int ) { const_iterator ci = *this; ++i; return ci; }
      const_iterator& operator--() { --i; return *this; }
      const_iterator operator--( int ) { const_iterator ci = *this; --i; return ci; }
      bool operator==( const const_iterator& ci ) const { 
	return keyRef_ == ci.keyRef_ && valRef_ == ci.valRef_ && i == ci.i; 
      }
      bool operator!=( const const_iterator& ci ) const { return i != ci.i; }
      KeyRef key() const { return KeyRef( keyRef_, i->first ); }
      typename Tag::val_type val() const {
	return Tag::val( valRef_, i->second() );
      }
      KeyVal operator *() const {
	return KeyVal( key(), val() );
      }
    private:
      KeyRefProd keyRef_;
      ValRefProd valRef_;
      typename map_type::const_iterator i;
    };

    /// first iterator over the map (read only)
    const_iterator begin() const { return const_iterator( keyRef_, valRef_, map_.begin() );  }
    /// last iterator over the map (read only)
    const_iterator end() const { return const_iterator( keyRef_, valRef_, map_.end() );  }
    /// find an entry in the map
    const_iterator find( const KeyRef & k ) const {
      checkKey( k );
      typename map_type::const_iterator f = map_.find( k.index() );
      return const_iterator( keyRef_, valRef_, f );
    }
    /// return element with key i
    KeyVal operator[]( size_type i ) const {
      typename map_type::const_iterator f = map_.find( k.index() );
      if ( f == map_.end() ) 
	throw edm::Exception( edm::errors::InvalidReference )
	  << "can't find reference in AssociationMap at position " << i;
      const_iterator ci( keyRef_, valRef_, f );
      return * ci;
    } 

  private:
    /// throw if k hasn't the same if as keyRef_
    void checkKey( const KeyRef & k ) const {
      if ( k.id() != keyRef_.id() )
	throw edm::Exception( edm::errors::InvalidReference, "invalid key reference" );
    }
    /// throw if v hasn't the same if as valRef_
    void checkVal( const ValRef & v ) const {
      if ( v.id() != valRef_.id() )
	throw edm::Exception( edm::errors::InvalidReference, "invalid value reference" );
    }
    /// reference to "key" collection
    KeyRefProd keyRef_;
    /// reference to "value" collection
    ValRefProd valRef_;
    /// index map
    map_type map_;
  };

}

#endif
