#ifndef RecoTracker_MkFitCore_interface_ConfigWrapper_h
#define RecoTracker_MkFitCore_interface_ConfigWrapper_h

namespace mkfit {
  /**
   * The purpose of this namespace is to hide the header of Config.h
   * from CMSSW. This header contain uses of the build-time
   * configuration macros, that should remain as internal details of
   * MkFit package.
   */
  namespace ConfigWrapper {
    void initializeForCMSSW();
  }  // namespace ConfigWrapper
}  // namespace mkfit

#endif
