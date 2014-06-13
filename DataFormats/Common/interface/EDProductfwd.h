#ifndef DataFormats_Common_EDProductfwd_h
#define DataFormats_Common_EDProductfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

----------------------------------------------------------------------*/
#include <memory>

#include "FWCore/Utilities/interface/HideStdSharedPtrFromRoot.h"

namespace edm 
{
  class BasicHandle;
  class WrapperHolder;
  class WrapperOwningHolder;
  class EDProductGetter;
  class ProductID;
  class OutputHandle;
  class RefCore;

  template <typename C, typename T, typename F> class Ref;
  template <typename T> class RefToBase;
  template <typename K> class RefBase;
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
    class RefVectorHolderBase;
    template <class T> class BaseVectorHolder;
    template <class T, class REFVECTOR> class VectorHolder;
  }
  typedef std::shared_ptr<reftobase::RefHolderBase> helper_ptr;
  typedef reftobase::RefVectorHolderBase helper_vector;
  typedef std::shared_ptr<reftobase::RefVectorHolderBase> helper_vector_ptr;
}

#endif
