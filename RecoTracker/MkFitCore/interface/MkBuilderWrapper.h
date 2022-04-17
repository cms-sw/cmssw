#ifndef RecoTracker_MkFitCore_interface_MkBuilderWrapper_h
#define RecoTracker_MkFitCore_interface_MkBuilderWrapper_h

#include <memory>

namespace mkfit {
  class MkBuilder;

  /**
   * The purpose of this class is to hide the header of MkBuilder.h
   * from CMSSW. The headers included by MkBuilder.h contain uses of
   * the build-time configuration macros, that should remain as
   * internal details of MkFit package.
   */
  class MkBuilderWrapper {
  public:
    MkBuilderWrapper(bool silent);
    ~MkBuilderWrapper();

    MkBuilder& get() { return *builder_; }

    static void populate();

  private:
    std::unique_ptr<MkBuilder> builder_;
  };
}  // namespace mkfit

#endif
