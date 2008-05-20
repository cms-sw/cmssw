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


#define CLASS_ID(type_, name_)  \
typedef cond::ClassID<type_> name_; \
namespace{ name_ EDM_PLUGIN_SYM(instance_cld, __LINE__); } \
int bha##name_
//DEFINE_EDM_PLUGIN(cond::ClassIdFactory, name_ , packageClassIDRegistry.csids.back())
  
