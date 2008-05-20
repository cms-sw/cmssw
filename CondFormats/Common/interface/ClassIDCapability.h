/*
 * no ifndef protection
 * this file is supposed to be included once, and only once per module
 *
 */

namespace cond {
  class ClassIDRegistry;
}
namespace {
  cond::ClassIDRegistry * =0;
}

#include "CondFormats/Common/interface/ClassIDRegistry.h"

namespace {
  cond::ClassIDRegistry packageClassIDRegistry("LCGClassID/");
}

ELEM_CONSTR(packageClassIDRegistry)

extern "C" void SEAL_CAPABILITIES (const char**& names, int& n )
{ 
  names = &packageClassIDRegistry.csids.front();
  n =  packageClassIDRegistry.csids.size();
}


