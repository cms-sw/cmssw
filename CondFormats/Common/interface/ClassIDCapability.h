/*
 * no ifndef protection
 * this file is supposed to be included once, and only once per module
 *
 */

#include "CondFormats/Common/interface/ClassIDRegistry.h"

namespace {
  cond::ClassIDRegistry packageClassIDRegistry(cond::idCategories::dictIDCategory);
}

ELEM_CONSTR(packageClassIDRegistry)


 
extern "C" void NOT_SEAL_CAPABILITIES (const char**& names, int& n )
{ 
  names = &packageClassIDRegistry.csids.front();
  n =  packageClassIDRegistry.csids.size();
}


#define CLASS_ID(type_)  \
namespace{ cond::ClassID<type_>  EDM_PLUGIN_SYM(instance_cld, __LINE__)(0); }	\
 DEFINE_EDM_PLUGIN(cond::ClassInfoFactory, cond::ClassID<type_>  , cond::ClassID<type_>().pluginName(cond::idCategories::dictIDCategory).c_str() )
  
