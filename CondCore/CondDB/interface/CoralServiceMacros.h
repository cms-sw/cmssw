#ifndef CondCore_CondDB_CoralServiceMacros_h
#define CondCore_CondDB_CoralServiceMacros_h

#include "CondCore/CondDB/interface/CoralServiceFactory.h"
#include "CoralKernel/Service.h"

#define DEFINE_CORALSERVICE(type,name) \
  DEFINE_EDM_PLUGIN (cond::CoralServicePluginFactory,type,name)

#define DEFINE_CORALSERVICE(type,name) \
  DEFINE_EDM_PLUGIN (cond::CoralServicePluginFactory,type,name)

#endif
