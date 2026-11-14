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

  // Sets the product getter used to read RefCores on this thread, and restores the previous one
  // when destroyed: guards can nest on the same thread, e.g. when a task that reads RefCores runs
  // while the thread waits inside another guard.
  class RefCoreStreamerGuard {
  public:
    RefCoreStreamerGuard(EDProductGetter const* ep) : previous_(setRefCoreStreamer(ep)) {}
    ~RefCoreStreamerGuard() { unsetRefCoreStreamer(previous_); }
    RefCoreStreamerGuard(RefCoreStreamerGuard const&) = delete;
    RefCoreStreamerGuard& operator=(RefCoreStreamerGuard const&) = delete;
    RefCoreStreamerGuard(RefCoreStreamerGuard&&) = delete;
    RefCoreStreamerGuard& operator=(RefCoreStreamerGuard&&) = delete;

  private:
    static void unsetRefCoreStreamer(EDProductGetter const* previous);
    static EDProductGetter const* setRefCoreStreamer(EDProductGetter const* ep);

    EDProductGetter const* previous_;
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
