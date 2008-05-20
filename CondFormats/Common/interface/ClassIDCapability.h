/*
 * no ifndef protection
 * this file is supposed to be included once, and only once per module
 *
 */

#include "CondFormats/Common/interface/ClassIDRegistry.h"

namespace {
  cond::ClassIDRegistry packageClassIDRegistry("LCGClassID/");
}

// ELEM_CONSTR(packageClassIDRegistry)
cond::ClassIDRegistry::Elem::Elem(){registry = &packageClassIDRegistry;}

 
extern "C" void NOT_SEAL_CAPABILITIES (const char**& names, int& n )
{ 
  names = &packageClassIDRegistry.csids.front();
  n =  packageClassIDRegistry.csids.size();
}


#define CLASS_ID(type)  \
static cond::ClassID<type> EDM_PLUGIN_SYM(instance_cld, __LINE__); \
DEFINE_EDM_PLUGIN(cond::ClassIDRegistry::Elem, cond::ClassID<type>, packageClassIDRegistry.csids.back())
  
