#ifndef Common_OneToManyAssociation_h
#define Common_OneToManyAssociation_h
/** \class edm::OneToManyAssociation
 *
 * one-to-many reference map using EDM references
 * 
 * \author Luca Lista, INFN
 *
 * $Id: OneToManyAssociation.h,v 1.2 2006/03/16 16:35:29 llista Exp $
 *
 */
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <map>
#include <vector>

namespace edm {
  template<typename CKey, typename CVal, typename index = unsigned long>
  struct OneToManyAssociation {
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// reference to "value" collection
    typedef edm::RefProd<CVal> ValRefProd;
    /// reference to an object in "key" collection
    typedef edm::Ref<CKey> KeyRef;
    /// reference to an object on "value" collection
    typedef edm::Ref<CVal> ValRef;
    /// reference to an object on "value" collection
    typedef edm::RefVector<CVal> ValRefVec;
    /// map type
    typedef  std::map<index, std::vector<index> > map_type;
    /// size type
    typedef typename map_type::size_type size_type;
    /// default constructor
    OneToManyAssociation() { }
    /// constructor from product references
    OneToManyAssociation( const KeyRef & k, const ValRef & v ) :
      keyRef_( k ), valRef_( v ) {
    }
    /// map size
    size_type size() const { return map_.size(); }
    /// return true if empty
    bool empty() const { return map_.empty(); }
    /// insert an association
    void insert( const KeyRef & k, const ValRef & v ) {
      checkKey( k ); checkVal( v );
      index ik = index( k.index() ), iv = index( v.index() );
      map_[ ik ].push_back( iv );
    }

    struct const_iterator {
      typedef ptrdiff_t difference_type;
      typedef typename map_type::const_iterator::iterator_category iterator_category;
      const_iterator() { }
      const_iterator( typename map_type::const_iterator mi ) : i ( mi ) { }
      const_iterator & operator=( const const_iterator & it ) { i = it.i; return *this; }
      const_iterator& operator++() { ++i; return *this; }
      const_iterator operator++( int ) { const_iterator ci = *this; ++i; return ci; }
      const_iterator& operator--() { --i; return *this; }
      const_iterator operator--( int ) { const_iterator ci = *this; --i; return ci; }
      bool operator==( const const_iterator& ci ) const { return i == ci.i; }
      bool operator!=( const const_iterator& ci ) const { return i != ci.i; }
      KeyRef key() const { return KeyRef( keyRef_, i->first ); }
      ValRefVec values() const {
	ValRefVec v( valRef_.id() );
	const std::vector<index> & val = i->second;
	for( typename std::vector<index>::const_iterator idx = val.begin(); idx != val.end(); ++ idx )
	  v.push_back( ValRef( valRef_, * idx ) );
	return v;
      }
    private:
      typename map_type::const_iterator i;
    };

    /// first iterator over the map (read only)
    const_iterator begin() const { return const_iterator( map_.begin() );  }
    /// last iterator over the map (read only)
    const_iterator end() const { return const_iterator( map_.end() );  }
    /// find an entry in the map
    const_iterator find( const KeyRef & k ) const {
      checkKey( k );
      typename map_type::const_iterator f = map_.find( k.index() );
      return const_iterator( f );
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
