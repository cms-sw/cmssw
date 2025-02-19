#ifndef CondCore_DBCommon_CoralServiceMacros_h
#define CondCore_DBCommon_CoralServiceMacros_h

#include "CondCore/DBCommon/interface/CoralServiceFactory.h"
#include "CoralKernel/Service.h"

#define DEFINE_CORALSERVICE(type,name) \
  DEFINE_EDM_PLUGIN (cond::CoralServicePluginFactory,type,name)

#define DEFINE_CORALSERVICE(type,name) \
  DEFINE_EDM_PLUGIN (cond::CoralServicePluginFactory,type,name)

#endif
