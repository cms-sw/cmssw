#ifndef Common_EDProductfwd_h
#define Common_EDProductfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

$Id: EDProductfwd.h,v 1.12 2007/06/07 15:40:40 paterno Exp $

----------------------------------------------------------------------*/

namespace edm 
{
  class BasicHandle;
  class EDProduct;
  class EDProductGetter;
  class ProductID;
  class RefCore;

  template <typename C, typename T, typename F> class Ref;
  template <typename T> class RefToBase;
  template <typename K> class RefBase;
  template <typename K> class RefItem;
  template <typename T> class RefProd;
  template <typename C, typename T, typename F> class RefVector;
  template <class T> class RefToBaseVector;
  template <typename T> class RefVectorBase;
  template <typename C, typename T, typename F> class RefVectorIterator;
  template <typename T> class Wrapper;
  template <typename T> class Handle;
  template <typename T> class OrphanHandle;

  namespace reftobase
  {
    template <typename T>  class BaseHolder;
    template <typename T, class REF> class Holder;
    class RefHolderBase;
    template <typename T> class IndirectHolder;
    template <typename REF> class RefHolder;

    template <class T> class BaseVectorHolder;
    template <class T, class REFVECTOR> class VectorHolder;
  }
}

#endif
