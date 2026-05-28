#ifndef FWCore_Integration_plugins_Require_h
#define FWCore_Integration_plugins_Require_h

#include "FWCore/Utilities/interface/Exception.h"

#define REQUIRE(cond)                                                                            \
  do {                                                                                           \
    if (!(cond))                                                                                 \
      throw cms::Exception("Assert") << "Assertion failed: " #cond " (line " << __LINE__ << ")"; \
  } while (false)

#endif
