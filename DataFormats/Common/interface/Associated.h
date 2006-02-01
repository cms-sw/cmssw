#ifndef Common_Associated_h
#define Common_Associated_h
#include "DataFormats/Common/interface/ExtCollection.h"
#include "FWCore/EDProduct/interface/Ref.h"
#include "FWCore/EDProduct/interface/RefProd.h"

namespace edm {
  
  template<typename C, typename T = typename C::value_type>
  struct Associated {
    template<typename CExt, typename Ext>
    static const T & get( const Ref<ExtCollection<CExt, Ext> > & ref, 
			  const RefProd<C> & ( Ext:: * getRef )() const,
			  size_t index
			  ) {
      return ( * (ref.product()->ext().* getRef)() )[ index ];
    }
    template<typename CExt, typename Ext>
    static const T & get( const Ref<ExtCollection<CExt, Ext> > & ref, 
			  const RefProd<C> & ( Ext:: * getRef )() const
			  ) {
      return get( ref, getRef, ref.index() );
    }
  };

}

#endif
