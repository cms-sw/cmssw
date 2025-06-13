#ifndef DataFormats_Common_RefCoreStreamer_h
#define DataFormats_Common_RefCoreStreamer_h

#include "TClassStreamer.h"
#include "TClassRef.h"

class TBuffer;

namespace edm {
  class EDProductGetter;
  class RefCoreStreamer : public TClassStreamer {
  public:
    explicit RefCoreStreamer() : cl_("edm::RefCore") {}

    void operator()(TBuffer& R__b, void* objp) override;

    TClassStreamer* Generate() const override;

  private:
    TClassRef cl_;
  };

  class RefCoreWithIndexStreamer : public TClassStreamer {
  public:
    explicit RefCoreWithIndexStreamer() : cl_("edm::RefCoreWithIndex") {}

    void operator()(TBuffer& R__b, void* objp) override;

    TClassStreamer* Generate() const override;

  private:
    TClassRef cl_;
  };

  void setRefCoreStreamerInTClass();

  class RefCoreStreamerGuard {
  public:
    RefCoreStreamerGuard(EDProductGetter const* ep) { setRefCoreStreamer(ep); }
    ~RefCoreStreamerGuard() { unsetRefCoreStreamer(); }
    RefCoreStreamerGuard(RefCoreStreamerGuard const&) = delete;
    RefCoreStreamerGuard& operator=(RefCoreStreamerGuard const&) = delete;
    RefCoreStreamerGuard(RefCoreStreamerGuard&&) = delete;
    RefCoreStreamerGuard& operator=(RefCoreStreamerGuard&&) = delete;

  private:
    static void unsetRefCoreStreamer();
    static EDProductGetter const* setRefCoreStreamer(EDProductGetter const* ep);
  };
  class MultiThreadRefCoreStreamerGuard {
  public:
    MultiThreadRefCoreStreamerGuard(EDProductGetter const* ep) { setRefCoreStreamer(ep); }
    ~MultiThreadRefCoreStreamerGuard() { unsetRefCoreStreamer(); }

  private:
    static void setRefCoreStreamer(EDProductGetter const* ep);
    static void unsetRefCoreStreamer();
  };

}  // namespace edm
#endif
