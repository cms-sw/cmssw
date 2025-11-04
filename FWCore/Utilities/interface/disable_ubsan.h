#ifndef FWCore_Utilites_disable_ubsan_h
#define FWCore_Utilites_disable_ubsan_h
// With gcc 13.4.0, some summary routines are failing with unreachable program point
// UBSAN errors.  No UB has been identified, so for now this workaround suppresses
// UBSAN checking for the routines that are failing.
//
// details at https://github.com/cms-sw/cmssw/issues/49151
#ifdef CMS_UNDEFINED_SANITIZER
#define DISABLE_UBSAN __attribute__((no_sanitize("undefined")))
#else
#define DISABLE_UBSAN
#endif
#endif
