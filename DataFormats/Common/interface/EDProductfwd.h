#ifndef Common_EDProductfwd_h
#define Common_EDProductfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

$Id: EDProductfwd.h,v 1.2 2006/01/06 00:25:27 wmtan Exp $

----------------------------------------------------------------------*/

namespace edm {
  class EDProduct;
  class EDProductGetter;
  class ParameterSetID;
  class ProductID;
  class RefBase;
  class RefVectorBase;
  class RefCore;
  class RefItem;

  template <typename T> class Wrapper;
  template <typename T> class RefProd;
  template <typename C, typename T> class Ref;
  template <typename C, typename T> class RefVector;
  template <typename C, typename T> class RefVectorIterator;
}

// The following are trivial enough so that the real headers can be included.
#include "DataFormats/Common/interface/EventID.h"

#endif
