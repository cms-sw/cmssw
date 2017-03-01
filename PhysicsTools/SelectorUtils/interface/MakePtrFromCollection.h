#ifndef __PhysicsTools_SelectorUtils_MakePtrFromCollection_h__
#define __PhysicsTools_SelectorUtils_MakePtrFromCollection_h__

#include "DataFormats/Common/interface/Ptr.h"

template<class Collection, class InPhysObj = typename Collection::value_type, class OutPhysObj = typename Collection::value_type>
struct MakePtrFromCollection{
  edm::Ptr<OutPhysObj> operator()(const Collection& coll, unsigned idx) {
    edm::Ptr<InPhysObj> temp(&coll,idx);
    return edm::Ptr<OutPhysObj>(temp);
  }
};

#endif
