#ifndef FWCore_Utilites_disable_ubsan_h
#define FWCore_Utilites_disable_ubsan_h
//
// function attribute to suppress UBSAN checking for routines that
// give false positives for undefined behavior
//
// background at https://github.com/cms-sw/cmssw/issues/49151
//
// usage: void f() DISABLE_UBSAN {}
//
#ifdef CMS_UNDEFINED_SANITIZER
#define DISABLE_UBSAN __attribute__((no_sanitize("undefined")))
#else
#define DISABLE_UBSAN
#endif
#endif
