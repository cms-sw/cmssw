#ifndef Common_getAssociated_h
#define Common_getAssociated_h
#include "DataFormats/Common/interface/ref_collection.h"

template<typename C, typename R>
const typename R::value_type & getAssociated( const  edm::Ref<ref_collection<C, R> > & ref ) {
  // the following could be a method of edm::Ref
  const edm::EDProduct * edp = ref.productGetter()->getIt( ref.id() );
  const edm::Wrapper<ref_collection<C, R> > * w = 
    dynamic_cast<const edm::Wrapper<ref_collection<C, R> > *>( edp );
  const ref_collection<C, R> * coll = w->product();
  // 
  edm::RefProd<R> assocCollRef = coll->ref();
  const typename R::value_type & ret = (*assocCollRef)[ ref.index() ];
  return ret;
}

#endif
