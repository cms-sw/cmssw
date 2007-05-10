#ifndef Framework_DelayedReader_h
#define Framework_DelayedReader_h

/*----------------------------------------------------------------------
  
DelayedReader: The abstract interface through which the EventPrincipal
uses input sources to retrieve EDProducts from external storage.

$Id: DelayedReader.h,v 1.4 2006/02/07 07:51:41 wmtan Exp $

----------------------------------------------------------------------*/

#include <memory>

namespace edm {
  class BranchKey;
  class BranchEntryDescription;
  class EDProduct;
  class EDProductGetter;
  class DelayedReader {
  public:
    virtual ~DelayedReader();

    virtual std::auto_ptr<EDProduct> getProduct(BranchKey const& k, EDProductGetter const* ep) const = 0;
    virtual std::auto_ptr<BranchEntryDescription> getProvenance(BranchKey const& k, EDProductGetter const* ep) const = 0;
  };
}

#endif
