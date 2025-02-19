#ifndef DataFormats_Common_RefCoreStreamer_h
#define DataFormats_Common_RefCoreStreamer_h

#include "TClassStreamer.h"
#include "TClassRef.h"
#include <assert.h>

class TBuffer;

namespace edm {
  class EDProductGetter;
  class RefCoreStreamer : public TClassStreamer {
  public:
    explicit RefCoreStreamer() : cl_("edm::RefCore"){}

    void operator() (TBuffer &R__b, void *objp);

  private:
    TClassRef cl_;
  };

  class RefCoreWithIndexStreamer : public TClassStreamer {
  public:
    explicit RefCoreWithIndexStreamer() : cl_("edm::RefCoreWithIndex"){}
    
    void operator() (TBuffer &R__b, void *objp);
    
  private:
    TClassRef cl_;
  };

  void setRefCoreStreamer(bool resetAll = false);
  EDProductGetter const* setRefCoreStreamer(EDProductGetter const* ep);
}
#endif
