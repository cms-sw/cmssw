#ifndef Common_OneToManyAssociation_h
#define Common_OneToManyAssociation_h
/** \class edm::OneToManyAssociation OneToManyAssociation.h DataFormats/Common/interface/OneToManyAssociation.h
 *
 * one-to-many reference map using EDM references
 * 
 * \author Luca Lista, INFN
 *
 * $Id: OneToManyAssociation.h,v 1.8 2006/04/20 08:50:24 llista Exp $
 *
 */
#include "DataFormats/Common/interface/AssociationMapBase.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "FWCore/Utilities/interface/EDMException.h"
#include <map>
#include <vector>

namespace edm {

  template<typename val_type, typename val_ref, typename index>
  struct AssociationConstIteratorHelper<std::map<index, std::vector<index> >, val_type, val_ref> {
    template<typename val_prod, typename idx_val>
    static val_type val( const val_prod & vp, const idx_val & i ) {
      val_type v( vp.id() );
      const std::vector<index> & val = i;
      for( typename std::vector<index>::const_iterator idx = val.begin(); idx != val.end(); ++ idx )
	v.push_back( val_ref( vp, * idx ) );
      return v;
    }
  };

  template<typename CKey, typename CVal, typename index = unsigned long>
  struct OneToManyAssociation : public AssociationMapBase<CKey, CVal, index> {
    /// base class
    typedef AssociationMapBase<CKey, CVal, index> base;
    /// reference to "key" collection
    typedef typename base::KeyRefProd KeyRefProd;
    /// reference to "value" collection
    typedef typename base::ValRefProd ValRefProd;
    /// reference to an object in "key" collection
    typedef typename base::KeyRef KeyRef;
    /// reference to an object on "value" collection
    typedef typename base::ValRef ValRef;
    /// reference to an object on "value" collection
    typedef edm::RefVector<CVal> ValRefVec;
    /// map type
    typedef  std::map<index, std::vector<index> > map_type;
    /// size type
    typedef typename map_type::size_type size_type;
    /// default constructor
    OneToManyAssociation() { }
    /// constructor from product references
    OneToManyAssociation( const KeyRefProd & k, const ValRefProd & v ) :
      base( k, v ) {
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
      map_[ ik ].push_back( iv );
    }
    /// key/values pair structure
    struct keyVal {
      typedef KeyRef key_type;
      typedef ValRefVec val_type;
      keyVal() { }
      keyVal( const key_type & k, const val_type & v ) : key( k ), values( v ) { }
      key_type key;
      val_type values;
    };
    /// define value type for this collection
    typedef keyVal value_type;
    /// define const_iterator
    typedef typename base::template const_iterator<map_type, keyVal> const_iterator;

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
    keyVal operator[]( size_type i ) const {
      typename map_type::const_iterator f = map_.find( k.index() );
      if ( f == map_.end() ) 
	throw edm::Exception( edm::errors::InvalidReference )
	  << "can't find reference in OneToManyAssociation at position " << i;
      const_iterator ci( keyRef_, valRef_, f );
      return * ci;
    } 

  private:
    /// index map
    map_type map_;
  };

}

#endif
