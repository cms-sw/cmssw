#ifndef Common_AssociationMapBase_h
#define Common_AssociationMapBase_h
/** \class edm::AssociationMapBase AssociationMapBase.h DataFormats/Common/interface/AssociationMapBase.h
 *
 * base class for OneToOneAssociation and OneToManyAssociation
 * 
 * \author Luca Lista, INFN
 *
 * $Id: OneToOneAssociation.h,v 1.2 2006/04/20 08:50:24 llista Exp $
 *
 */
#include "DataFormats/Common/interface/RefProd.h"
#include "DataFormats/Common/interface/Ref.h"

namespace edm {

  template<typename map_type, typename val_type, typename val_ref>
  struct AssociationConstIteratorHelper {
    template<typename val_prod, typename idx_val>
    static val_type val( const val_prod &, const idx_val & );
  };

  template<typename CKey, typename CVal, typename index = unsigned long>
  class AssociationMapBase {
  public:
    /// reference to "key" collection
    typedef edm::RefProd<CKey> KeyRefProd;
    /// reference to "value" collection
    typedef edm::RefProd<CVal> ValRefProd;
    /// reference to an object in "key" collection
    typedef edm::Ref<CKey> KeyRef;
    /// reference to an object on "value" collection
    typedef edm::Ref<CVal> ValRef;
    /// default constructor
    AssociationMapBase() { }
    /// constructor from product references
    AssociationMapBase( const KeyRefProd & k, const ValRefProd & v ) :
      keyRef_( k ), valRef_( v ) {
    }
  protected:
    /// const_iterator base template
    template<typename map_type, typename keyVal>
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
      typedef typename keyVal::key_type key_type;
      typedef typename keyVal::val_type val_type;
      key_type key() const { return key_type( keyRef_, i->first ); }
      val_type val() const { return AssociationConstIteratorHelper<map_type, val_type, ValRef>::val( valRef_, i->second ); }
      keyVal operator *() const {
	return keyVal( key(), values() );
      }
    private:
      KeyRefProd keyRef_;
      ValRefProd valRef_;
      typename map_type::const_iterator i;
    };

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
  };

}

#endif
