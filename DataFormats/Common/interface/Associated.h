#ifndef Common_Associated_h
#define Common_Associated_h
#include "DataFormats/Common/interface/ext_collection.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefProd.h"

template<typename C, typename T = typename C::value_type>
struct Associated {
  template<typename CExt, typename Ext>
  static const T & get( const edm::Ref<ext_collection<CExt, Ext> > & ref, 
				 const edm::RefProd<C> & ( Ext:: * getRef )() const ) {
    return ( * (ref.product()->ext().* getRef)() )[ ref.index() ];
  }
};

#endif
