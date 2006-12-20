#ifndef Common_EDProductfwd_h
#define Common_EDProductfwd_h

/*----------------------------------------------------------------------
  
Forward declarations of types in the EDM.

$Id: EDProductfwd.h,v 1.5 2006/12/05 23:56:17 paterno Exp $

----------------------------------------------------------------------*/

#include "DataFormats/Common/interface/ModuleDescriptionID.h"
#include "DataFormats/Common/interface/ParameterSetID.h"
#include "DataFormats/Common/interface/ProcessHistoryID.h"
#include "DataFormats/Common/interface/ProcessConfigurationID.h"

namespace edm 
{
  class EDProduct;
  class EDProductGetter;
  class ProductID;
  class RefCore;
  class Timestamp;
  class EventID;

  struct BranchDescription;
  struct ModuleDescription;

  template <typename C, typename T, typename F> class Ref;
  template <typename T> class RefBase;
  template <typename T> class RefItem;
  template <typename T> class RefProd;
  template <typename C, typename T, typename F> class RefVector;
  template <typename T> class RefVectorBase;
  template <typename C, typename T, typename F> class RefVectorIterator;
  template <typename T> class Wrapper;
}

// The following are trivial enough so that the real headers can be included.
#include "DataFormats/Common/interface/EventID.h"

#endif
