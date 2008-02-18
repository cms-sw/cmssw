#ifndef IOPool_Common_RefStreamer_h
#define IOPool_Common_RefStreamer_h

#include "TClassStreamer.h"
#include "TClassRef.h"
#include <assert.h>

class TBuffer;

namespace edm {
  class EDProductGetter;
  class RefStreamer : public TClassStreamer {
  public:
    explicit RefStreamer(EDProductGetter const* ep) : cl_("edm::RefCore"), prodGetter_(ep) {}

    void setProductGetter(EDProductGetter const* ep) {
	assert(ep);
	prodGetter_ = ep;
    }
    void operator() (TBuffer &R__b, void *objp);

  private:
    TClassRef cl_;
    EDProductGetter const* prodGetter_;
  };

  void SetRefStreamer();
  void SetRefStreamer(EDProductGetter const* ep);
}

#endif
