#ifndef RefMacros_h
#define RefMacros_h

#include <vector>
#include "DataFormats/Common/interface/Ref.h"
#include "DataFormats/Common/interface/FwdRef.h"
#include "DataFormats/Common/interface/RefVector.h"
#include "DataFormats/Common/interface/RefProd.h"

#define DECLARE_EDM_REFS( class_name )                                          \
  typedef std::vector< class_name > class_name ## Collection;                   \
  typedef edm::Ref< class_name ## Collection>       class_name ## Ref;          \
  typedef edm::FwdRef< class_name ## Collection>       class_name ## FwdRef;    \
  typedef edm::RefProd< class_name ## Collection>   class_name ## RefProd;      \
  typedef edm::RefVector< class_name ## Collection> class_name ## RefVector;
      
#endif // RefMacros_h
