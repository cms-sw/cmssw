#ifndef FWCore_FWLite_RefStreamer_h
#define FWCore_FWLite_RefStreamer_h

#include "TClassStreamer.h"
#include "TClassRef.h"
class TBuffer;

namespace edm {
  class EDProductGetter;
}  
namespace fwlite {
  class RefStreamer : public TClassStreamer {
  public:
    explicit RefStreamer(edm::EDProductGetter const* ep) : cl_("edm::RefCore::RefCoreTransients"), prodGetter_(ep) {}

    edm::EDProductGetter const* setProductGetter(edm::EDProductGetter const* ep) {
      edm::EDProductGetter const* previous = prodGetter_;
      prodGetter_ = ep;
      return previous;
    }
    void operator() (TBuffer &R__b, void *objp);

  private:
    TClassRef cl_;
    edm::EDProductGetter const* prodGetter_;
  };

}

#endif
